import numpy as np
import torch
import gc
import torch.nn as nn
from sklearn.model_selection import KFold
from data.dataset import SddpDataset
from data.sampler import SubsetSequentialSampler
from torch.utils.data import DataLoader, SubsetRandomSampler
from network.transformer import Transformer
from train import fit, predict
from scheduler import CosineWarmupScheduler, CosineAnnealingWarmUpRestarts
from visualization.cuts_graph import *
from visualization.attention_score_graph import read_plot_alignment_matrices
from utils.sample_data import get_sample_data, get_max_cut_cnt_from_prediction

import os
import json
import warnings
import pickle
from datetime import datetime
import argparse
from tensorboardX import SummaryWriter

# 경고메세지 끄기
warnings.filterwarnings(action='ignore')


def main(args):
    gc.collect()
    torch.cuda.empty_cache()

    # configuration
    print("torch.cuda.is_available(): ", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.prob == "ProductionPlanning":
        src_dim = 57
        tgt_dim = 14  # 11
    elif args.prob == "EnergyPlanning":
        src_dim = 19  # 27
        tgt_dim = 7  # 9
    elif args.prob == "MertonsPortfolioOptimization":
        src_dim = 19  # 27
        tgt_dim = 9
    else:
        raise NotImplementedError

    # 데이터 불러오기
    if args.mode == 'train':
        save_path = os.path.join(args.save_path, "{}/stages_{}/train".format(args.prob,
                                                                                args.num_stages))  # /original/except_outliers
        if args.outlier == 'except_outlier':
            save_path = os.path.join(save_path, "except_outliers")

        if args.loss == 'MSE_CE':
            save_path = os.path.join(save_path, "change_loss")
        else:
            raise ValueError

    elif args.mode == 'inference':
        save_path = os.path.join(args.save_path,
                                 "{}/stages_{}/predict".format(args.prob, args.num_stages))  # /original/except_outliers

    if args.mode == 'inference_one_sample':
        x_raw_data, y_raw_data = get_sample_data(args, 20, 5)
    else:
        print(f"Data from {save_path}")
        with open(os.path.join(save_path, "features.pickle"), "rb") as fr:
            x_raw_data = pickle.load(fr)

        with open(os.path.join(save_path, "labels.pickle"), "rb") as fr:
            y_raw_data = pickle.load(fr)

    if args.mode == 'train':
        train(args, device, src_dim, tgt_dim, x_raw_data, y_raw_data)
    elif args.mode == 'inference':
        inference(args, device, src_dim, tgt_dim, x_raw_data, y_raw_data)
    elif args.mode == 'inference_one_sample':
        inference_one_sample(args, device, src_dim, tgt_dim, x_raw_data, y_raw_data, 1)
    else:
        raise ValueError


def train(args, device, src_dim, tgt_dim, x_raw_data, y_raw_data):
    splits = KFold(n_splits=args.kfold, shuffle=False)
    run_time = (datetime.now()).strftime("%Y%m%d_%H%M%S")
    dataset = SddpDataset(x_raw_data, y_raw_data)
    loss_fn = (nn.MSELoss(), nn.CrossEntropyLoss())
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        train_sampler = SubsetSequentialSampler(train_idx)
        val_sampler = SubsetSequentialSampler(val_idx)

        train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler)
        val_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler)

        model, optimizer, lr_scheduler = model_initialize(src_dim, tgt_dim, device, args)

        # log 설정 (tensorboard)
        log_path = os.path.join(os.getcwd(), "scripts", "logs", run_time + ", Fold {}".format(fold + 1))
        os.makedirs(log_path, exist_ok=True)

        if args:
            with open(os.path.join(log_path, "config.txt"), "w") as f:
                json.dump(args.__dict__, f, indent=4)

        writer = SummaryWriter(logdir=log_path)

        print("run: {}, Fold: {}".format(run_time, fold + 1))

        fit(model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_fn=loss_fn,
            tgt_train_raw_data=list(map(lambda i: y_raw_data[i], train_idx)),
            tgt_eval_raw_data=list(map(lambda i: y_raw_data[i], val_idx)),
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            epochs=args.epochs,
            fold=fold,
            args=args,
            writer=writer)


def inference(args, device, src_dim, tgt_dim, x_raw_data, y_raw_data):
    inference_error_ratio_pred = []
    inference_error_ratio_sddp = []
    for fold in range(1, 7):
        model, optimizer, lr_scheduler = model_initialize(src_dim, tgt_dim, device, args)

        if args.load_model == 'None':
            raise FileNotFoundError
        else:
            # Load model
            inf_path = os.path.join(os.getcwd(), "scripts", "logs", "{}, Fold {}".format(args.load_model, fold))
            model.load_state_dict(torch.load(os.path.join(inf_path, "best_episode.ckpt")))

        test_dataset = SddpDataset(x_raw_data, y_raw_data)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

        errors_pred, errors_sddp, pred_cut_ex, _, _, _ = predict(model=model,
                                                                 dataloader=test_dataloader,
                                                                 args=args,
                                                                 cnt_cuts=1,
                                                                 device=device)
        print(f"Fold {fold}, errors mean(sddp-transformer): ", errors_pred)

        inference_error_ratio_pred.append(errors_pred)

        print(f"Fold {fold}, errors mean(sddp): ", errors_sddp)

        inference_error_ratio_sddp.append(errors_sddp)

        if args.prob == "ProductionPlanning":
            num_var = 3
        elif args.prob == "EnergyPlanning":
            num_var = 1
        elif args.prob == "MertonsPortfolioOptimization":
            num_var = 2
        else:
            raise NotImplementedError
        for i in range(num_var):
            get_cut_graph(target_cut=y_raw_data[:args.num_stages - 1],
                          pred_cut=pred_cut_ex,
                          var_idx=i,
                          args=args,
                          save_path=inf_path)
    print("mean error ratio(sddp-transformer): ", np.mean(inference_error_ratio_pred))
    print("mean error ratio(sddp): ", np.mean(inference_error_ratio_sddp))


def inference_one_sample(args, device, src_dim, tgt_dim, x_raw_data, y_raw_data, fold):
    model, optimizer, lr_scheduler = model_initialize(src_dim, tgt_dim, device, args)

    if args.load_model == 'None':
        raise FileNotFoundError
    else:
        # Load model
        inf_path = os.path.join(os.getcwd(), "scripts", "logs", "{}, Fold {}".format(args.load_model, fold))
        model.load_state_dict(torch.load(os.path.join(inf_path, "best_episode.ckpt")))

    test_dataset = SddpDataset(x_raw_data, y_raw_data)
    test_dataloader = DataLoader(test_dataset, batch_size=args.num_stages)

    _, _, pred_cut_ex, encoder_weights, decoder_weights_sa, decoder_weights_mha = predict(model=model,
                                                                                          dataloader=test_dataloader,
                                                                                          args=args,
                                                                                          cnt_cuts=1,
                                                                                          device=device)

    # with open("D:/sddp_data/EnergyPlanning/stages_7/sample_scenario/cuts.pickle", 'rb') as fr:
    #     target_cuts = pickle.load(fr)
    #
    # get_sample_scenario_cuts_graph(target_cuts, pred_cut_ex[4][1])

    # print(encoder_weights)
    # decoder_weights_mha = decoder_weights_mha[:10]

    # with open(os.path.join('D:/sddp_data/EnergyPlanning/stages_7/sample/predict', "pred_cuts.pickle"), "wb") as fw:
    #     pickle.dump(pred_cut_ex, fw)

    print(pred_cut_ex.shape)

    size_reduced = 22

    get_max_cut_cnt_from_prediction(pred_cut_ex[0][:size_reduced, :size_reduced], problem='EnergyPlanning', n_stages=7)

    # decoder_weights_sa = torch.tril(decoder_weights_sa[:size_reduced+1, :size_reduced], diagonal=-1)[1:size_reduced+1]
    # decoder_weights_sa = decoder_weights_sa / torch.sum(decoder_weights_sa, dim=1, keepdim=True)
    decoder_weights_sa = decoder_weights_sa[:size_reduced, :size_reduced]

    read_plot_alignment_matrices(source_labels=np.arange(decoder_weights_sa.shape[1]),
                                 target_labels=np.arange(decoder_weights_sa.shape[0]),
                                 alpha=decoder_weights_sa)

    if args.prob == "ProductionPlanning":
        num_var = 3
    elif args.prob == "EnergyPlanning":
        num_var = 1
    elif args.prob == "MertonsPortfolioOptimization":
        num_var = 2
    else:
        raise NotImplementedError
    for i in range(num_var):
        get_cut_graph(target_cut=y_raw_data[:args.num_stages - 1],
                      pred_cut=pred_cut_ex,
                      var_idx=i,
                      args=args,
                      save_path=inf_path)


def model_initialize(src_dim, tgt_dim, device, args):
    model = Transformer(src_dim=src_dim,
                        tgt_dim=tgt_dim,
                        d_model=512,
                        nhead=8,
                        num_encoder_layers=6,
                        num_decoder_layers=6,
                        dropout=0.1,
                        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)  # lr=1e-8
    if args.lr_scheduler == 'CosineScheduler':
        lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=50, max_iters=2000)
    elif args.lr_scheduler == 'CosineAnnealingWarmRestarts':
        lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_max=args.eta_max,
                                                     T_up=args.T_up,
                                                     gamma=args.gamma)  # T_0=1700, T_mult=2, eta_max=1e-4, T_up=400, gamma=0.9
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1700, T_mult=2, eta_min=1e-5)
    elif args.lr_scheduler == 'None':
        lr_scheduler = None
    else:
        raise NotImplementedError
    return model, optimizer, lr_scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--prob', type=str, default='ProductionPlanning',
                        help='problem to solve')
    parser.add_argument('--num_stages', type=int, default=7,
                        help='Number of Stages')
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--kfold', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-8)
    parser.add_argument('--eta_max', type=float, default=1e-4)
    parser.add_argument('--T_0', type=int, default=4)
    parser.add_argument('--T_mult', type=int, default=2)
    parser.add_argument('--T_up', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str, default='None',
                        choices=['None', 'CosineScheduler', 'CosineAnnealingWarmRestarts'])
    parser.add_argument('--save_path', type=str, default='D:/sddp_data')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference', 'inference_one_sample'])
    parser.add_argument('--load_model', type=str, default='None')
    parser.add_argument('--feature_type', type=str, default='objective_information',
                        choices=['objective_information', 'no_objective_information'])
    parser.add_argument('--outlier', type=str, default='not_except_outlier',
                        choices=['not_except_outlier', 'except_outlier'])
    parser.add_argument('--loss', type=str, default='MSE',
                        choices=['MSE', 'MSE_CE'])
    main(parser.parse_args())
