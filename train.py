import numpy as np
import torch
import os
from envs.utils import get_parameters
from config import *
import cvxpy as cp

from visualization.cuts_graph import get_cut_graph


def fit(model, optimizer, lr_scheduler, loss_fn, tgt_train_raw_data, tgt_eval_raw_data, train_dataloader,
        val_dataloader, device, epochs, fold, args, writer):
    print("Training and validating model")
    for epoch in range(epochs):
        print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)

        train_loss, avg_error_train = train_loop(model, optimizer, lr_scheduler, loss_fn,
                                                                           tgt_train_raw_data, train_dataloader, args,
                                                                           device)
        validation_loss, avg_error_eval = validation_loop(model, loss_fn, tgt_eval_raw_data,
                                                                                   val_dataloader, epoch + 1, fold, args,
                                                                                   device)
        if lr_scheduler:
            lr_scheduler.step()

        writer.add_scalar(
            "Train Loss/per epoch",
            train_loss,
            epoch
        )

        writer.add_scalar(
            "Validation Loss/per epoch",
            validation_loss,
            epoch
        )

        writer.add_scalar(
            "Info/learning rate",
            optimizer.param_groups[0]['lr'],
            epoch
        )
        writer.add_scalar(
            "Avg Error(train)/per epoch",
            avg_error_train,
            epoch
        )
        writer.add_scalar(
            "Avg Error(eval)/per epoch",
            avg_error_eval,
            epoch
        )

        # writer.add_scalar(
        #     "Avg Error True End(train)/per epoch",
        #     avg_error_train_true_end,
        #     epoch
        # )
        # writer.add_scalar(
        #     "Avg Error True End(eval)/per epoch",
        #     avg_error_eval_true_end,
        #     epoch
        # )

        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Avg Error train: {avg_error_train:.4f}")
        print(f"Avg Error eval: {avg_error_eval:.4f}")
        # print(f"Avg Error True End train: {avg_error_train_true_end:.4f}")
        # print(f"Avg Error True End eval: {avg_error_eval_true_end:.4f}")
        print(f"learning rate: {optimizer.param_groups[0]['lr']}")
        print()

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(writer.logdir, "best_episode.ckpt"))


def train_loop(model, optimizer, lr_scheduler, loss_fn, tgt_raw_data, dataloader, args, device="cpu"):
    model.train()
    total_loss = 0
    error_rate = []

    temp = 0
    for batch in dataloader:
        X, y = batch[0], batch[1]
        X = X.to(device)
        y = y.to(device)

        y_input = y[:, :-1]
        y_answer = y[:, 1:]

        # target masking
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        # target padding
        y_raw = tgt_raw_data[temp:temp + y_input.size(0)]
        tgt_pad_mask = model.get_pad_mask(max_seq_len=sequence_length,
                                          data=y_raw).to(device)
        temp += y_input.size(0)

        # X, y_input, tgt_mask 인자로 전달하여 prediction 값 도출
        y_pred = model(X, y_input, tgt_mask, tgt_pad_mask=tgt_pad_mask)

        loss = loss_fn(y_pred, y_answer)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()

        end_token_idx = get_end_token_idx(y_pred[0], device) + 1  # get_end_token_idx(y_pred[0], device)
        # end_token_idx_true = y_raw[0].shape[0]  # y_raw[0].shape[0]-1

        obj_target = get_pred_obj(y_raw[0], args)  # y_raw[0][:-1]
        obj_pred = get_pred_obj(y_pred[0][:end_token_idx].detach().cpu().data.numpy(), args)
        # print("obj_target: ", obj_target)
        # print("obj_pred: ", obj_pred)
        error_rate.append(get_error_rate(obj_pred, obj_target) / np.abs(obj_target))

        # obj_pred_true = get_pred_obj(y_pred[0][:end_token_idx_true].detach().cpu().data.numpy(), args.prob)

        # error_rate_true_end.append(get_error_rate(obj_pred_true, obj_target) / obj_target)

    return total_loss / len(dataloader), np.mean(error_rate) #, np.mean(error_rate_true_end)


def validation_loop(model, loss_fn, tgt_raw_data, dataloader, epoch, fold, args, device="cpu"):
    model.eval()
    total_loss = 0
    error_rate = []
    error_rate_true_end = []

    temp = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            X, y = batch[0], batch[1]
            X = X.to(device)
            y = y.to(device)

            y_input = y[:, :-1]
            y_answer = y[:, 1:]

            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            # target padding
            y_raw = tgt_raw_data[temp:temp + y_input.size(0)]
            tgt_pad_mask = model.get_pad_mask(max_seq_len=sequence_length,
                                              data=y_raw).to(device)

            temp += y_input.size(0)

            y_pred = model(X, y_input, tgt_mask, tgt_pad_mask=tgt_pad_mask)

            if (epoch % 20 == 0 or epoch == 1) and idx == 0:
                get_predict_and_inference_cut_graph(X, y, y_pred, y_raw, model, epoch, fold, args, device)

            loss = loss_fn(y_pred, y_answer)
            total_loss += loss.detach().item()

            end_token_idx = get_end_token_idx(y_pred[0], device) + 1  # get_end_token_idx(y_pred[0], device)
            # end_token_idx_true = y_raw[0].shape[0]  # y_raw[0].shape[0]-1

            obj_target = get_pred_obj(y_raw[0], args)  # y_raw[0][:-1]

            obj_pred = get_pred_obj(y_pred[0][:end_token_idx].detach().cpu().data.numpy(), args)
            error_rate.append(get_error_rate(obj_pred, obj_target) / np.abs(obj_target))

            # obj_pred_true = get_pred_obj(y_pred[0][:end_token_idx_true].detach().cpu().data.numpy(),
            #                              args)

            # error_rate_true_end.append(get_error_rate(obj_pred_true, obj_target) / obj_target)

    return total_loss / len(dataloader), np.mean(error_rate) #, np.mean(error_rate_true_end)


def get_predict_and_inference_cut_graph(X, y, y_pred, y_raw, model, epoch, fold, args, device):

    if args.prob == "EnergyPlanning":
        max_length = 80
    elif args.prob == "MertonsPortfolioOptimization":
        max_length = 60
    else:
        max_length = 100
    y_inf, _, _, _ = get_pred_cuts(X, y, 1, model, device, max_length)
    inf_cut_ex = get_all_stage_cuts(y_inf, args, device)

    os.makedirs("D:/sddp_data/{}/{}/{}/{}/cuts/fold{}/inference/{}".format(args.prob, args.num_stages, args.mode, args.load_model, fold + 1, epoch),
                exist_ok=True)
    os.makedirs("D:/sddp_data/{}/{}/{}/{}/cuts/fold{}/predict/{}".format(args.prob, args.num_stages, args.mode, args.load_model, fold + 1, epoch),
                exist_ok=True)
    if args.prob == "ProductionPlanning":
        num_var = 3
    elif args.prob == "EnergyPlanning":
        num_var = 1
    elif args.prob == "MertonsPortfolioOptimization":
        num_var = 2
    else:
        raise NotImplementedError

    for i in range(num_var):
        get_cut_graph(target_cut=y_raw[:args.num_stages - 1],
                      pred_cut=inf_cut_ex,
                      var_idx=i,
                      args=args,
                      save_path="D:/sddp_data/{}/{}/{}/{}/cuts/fold{}/inference/{}".format(args.prob, args.num_stages,
                                                                                    args.mode, args.load_model, fold + 1, epoch))
    pred_cut_ex = get_all_stage_cuts(y_pred, args, device)
    for i in range(num_var):
        get_cut_graph(target_cut=y_raw[:args.num_stages - 1],
                      pred_cut=pred_cut_ex,
                      var_idx=i,
                      args=args,
                      save_path="D:/sddp_data/{}/{}/{}/{}/cuts/fold{}/predict/{}".format(args.prob, args.num_stages, args.mode,
                                                                                  args.load_model, fold + 1, epoch))
    print("epoch {} (fold{}): cuts figure saved".format(epoch, fold+1))


def predict(model, dataloader, args, cnt_cuts=2, device="cpu"):
    model.eval()
    pred_cut_ex = []
    errors = []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            X, y = batch[0], batch[1]
            error, pred_cut, encoder_weights, decoder_weights_sa, decoder_weights_mha = predict_one_batch(X, y, idx, model, args, cnt_cuts, device)
            errors += error
            if idx == 0:
                pred_cut_ex = pred_cut
    return np.mean(errors), pred_cut_ex, encoder_weights, decoder_weights_sa, decoder_weights_mha


def predict_one_batch(X, y, idx, model, args, cnt_cuts=2, device="cpu"):
    X = X.to(device)
    y = y.to(device)

    if args.prob == "EnergyPlanning":
        max_length = 80
    elif args.prob == "MertonsPortfolioOptimization":
        max_length = 60
    else:
        max_length = 100

    y_input, encoder_weights, decoder_weights_sa, decoder_weights_mha = get_pred_cuts(X, y, cnt_cuts, model, device, max_length)

    if idx == 0:
        pred_cut_ex = get_all_stage_cuts(y_input, args, device)
    else:
        pred_cut_ex = []

    errors = []
    for d in range(0, X.shape[0], 6):
        end_token_idx = get_end_token_idx(y_input[d], device) + 1
        obj_target = get_pred_obj(y[d].detach().cpu().data.numpy(), args)  # y_raw[0][:-1]

        obj_pred = get_pred_obj(y_input[d][:end_token_idx].detach().cpu().data.numpy(), args)
        print("obj_target", obj_target)
        print("obj_pred", obj_pred)
        errors.append(get_error_rate(obj_pred, obj_target) / np.abs(obj_target))
    return errors, pred_cut_ex, encoder_weights[end_token_idx-2][-1][0], decoder_weights_sa[end_token_idx-2][-1][0], decoder_weights_mha[end_token_idx-2][-1][0]


def get_pred_obj(cuts, args):
    A, B, b, c = get_parameters(args)

    if args.prob == "ProductionPlanning":
        num_Equality = 3
    elif args.prob == "EnergyPlanning":
        num_Equality = 2
    elif args.prob == "MertonsPortfolioOptimization":
        num_Equality = 2
    else:
        raise ValueError

    # cuts = cuts.detach().cpu().data.numpy()
    A = np.concatenate((A, np.zeros((A.shape[0], 1))), axis=1)
    # x = cp.Variable(shape=A.shape[1], nonneg=True)
    x = cp.Variable(shape=A.shape[1])

    A = np.concatenate((A, cuts[:, :-4]), axis=0)  # -1
    b = np.concatenate((b, cuts[:, -4].reshape(cuts.shape[0], 1)), axis=0)
    c = np.concatenate((c, np.array([1])), axis=0)

    constraints = [A[:num_Equality] @ x + np.squeeze(b[:num_Equality]) == 0] + [A[num_Equality:] @ x + np.squeeze(b[num_Equality:]) <= 0]
    if args.prob == "EnergyPlanning":
        constraints.append(x[:4] >= 0)
        obj = cp.Minimize(cp.exp(c[2]*x[1]+c[3]) + c[0]*x[2] + c[1]*x[3] + c[-1]*x[-1])
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.MOSEK)
    elif args.prob == "MertonsPortfolioOptimization":
        constraints.append(x[:4] >= 0)
        if utility_risk_aversion_coeff == 1:
            obj = cp.Minimize(-BigM * cp.log(x[2] + smallm) + x[4])
        else:
            obj = cp.Minimize(-1 * (1 / (1 - utility_risk_aversion_coeff) * cp.power(x[2], 1 - utility_risk_aversion_coeff)) + x[4])
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.MOSEK)
    else:
        obj = cp.Minimize(c @ x)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.CPLEX)
    # print("solution: {}".format(np.round(np.array(x.value), 2)))
    if prob.value == np.inf:
        return 2
    else:
        return prob.value


def get_error_rate(pred, target):
    return np.abs(pred - target)


def get_end_token_idx(cuts_pred, device="cpu"):
    # categorical token
    # end_token_idx = torch.argmax(cuts_pred[:, -1])
    token_idx = torch.argmax(cuts_pred[:, -3:], dim=1)
    try:
        end_token_idx = (token_idx == 2).nonzero(as_tuple=False)[0][0]
    except:
        end_token_idx = torch.tensor(cuts_pred.shape[0] - 1)
    return end_token_idx


def get_all_stage_cuts(y, args, device):
    cuts = []
    for i in range(args.num_stages - 1):
        cuts.append(y[i][:get_end_token_idx(y[i], device) + 1].detach().cpu().data.numpy())
    return cuts


def get_pred_cuts(X, y, cnt_cuts, model, device, max_length=100):
    y_input = y[:, :cnt_cuts]
    encoder_weights = []
    decoder_weights_sa = []
    decoder_weights_mha = []

    for _ in range(max_length):
        # Get source mask
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)

        y_pred, encoder_weight, decoder_weight_sa, decoder_weight_mha = model(X, y_input, tgt_mask)
        encoder_weights.append(encoder_weight)
        decoder_weights_sa.append(decoder_weight_sa)
        decoder_weights_mha.append(decoder_weight_mha)
        next_item = torch.unsqueeze(y_pred[:, -1], 1)

        # concatenate previous input with predicted cut
        y_input = torch.concat((y_input, next_item), dim=1)

    return y_input, encoder_weights, decoder_weights_sa, decoder_weights_mha

