import pickle
import torch
import os
from envs.utils import get_parameters
from config import *
from run_MSP import *



def fit(model, optimizer, lr_scheduler, loss_fn, tgt_train_raw_data, tgt_eval_raw_data, train_dataloader,
        val_dataloader, device, epochs, args, writer):
    print("Training and validating model")
    for epoch in range(epochs):
        print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)

        train_loss, avg_error_train, infeasible_cnt_ratio_train = train_loop(model, optimizer, loss_fn, tgt_train_raw_data,
                                                                             train_dataloader, args, device)
        validation_loss, avg_error_eval, infeasible_cnt_ratio_val = validation_loop(model, loss_fn, tgt_eval_raw_data,
                                                                                    val_dataloader, args, device)
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

        writer.add_scalar(
            "Infeasible Ratio Train/per epoch",
            infeasible_cnt_ratio_train,
            epoch
        )

        writer.add_scalar(
            "Infeasible Ratio Validation/per epoch",
            infeasible_cnt_ratio_val,
            epoch
        )

        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Avg Error train: {avg_error_train:.4f}")
        print(f"Avg Error eval: {avg_error_eval:.4f}")
        print(f"learning rate: {optimizer.param_groups[0]['lr']}")
        print(f"infeasible_cnt_ratio_train: {infeasible_cnt_ratio_train}")
        print(f"infeasible_cnt_ratio_val: {infeasible_cnt_ratio_val}")
        print()

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(writer.logdir, "best_episode.ckpt"))
    torch.save(model.state_dict(), os.path.join(writer.logdir, "best_episode.ckpt"))


def train_loop(model, optimizer, loss_fn, tgt_raw_data, dataloader, args, device="cpu"):
    if args.prob == "EnergyPlanning":
        if args.num_stages == 7:
            max_length = 80
        else:
            max_length = 90
    elif args.prob == "MertonsPortfolioOptimization":
        if args.num_stages == 7:
            max_length = 40
        else:
            max_length = 40
    else:
        max_length = 100
    model.train()
    total_loss = 0
    error_rate = []
    infeasible_cnt = 0
    infeasible_test_total_cnt = 0
    temp = 0
    for idx, batch in enumerate(dataloader):
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

        y_pred, _, _, _ = model(X, y_input, tgt_mask, tgt_pad_mask=tgt_pad_mask)

        loss = loss_fn[0](y_pred[:, :, :-4], y_answer[:, :, :-1]) + loss_fn[1](y_pred[:, :, -4:].transpose(1, 2),
                                                                               y_answer[:, :, -1].to(torch.long))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()

        if idx % 50 == 0:
            y_infer, _, _, _ = get_pred_cuts(X, y, 1, model, device, max_length)
            end_token_idx = get_end_token_idx(y_infer[0]) + 1
            obj_target = get_pred_obj(y[0][:, :-1].detach().cpu().data.numpy(), args)  # y_raw[0][:-1]

            obj_pred = get_pred_obj(y_infer[0][:end_token_idx, :-1].detach().cpu().data.numpy(), args)
            infeasible_test_total_cnt += 1
            if obj_pred == 0:
                infeasible_cnt += 1

            # print("obj_target: ", obj_target)
            # print("obj_pred: ", obj_pred)
            error_rate.append(get_error_rate(obj_pred, obj_target) / np.abs(obj_target))
            a = 0

    return total_loss / len(dataloader), np.mean(error_rate), infeasible_cnt / infeasible_test_total_cnt


def validation_loop(model, loss_fn, tgt_raw_data, dataloader, args, device="cpu"):
    if args.prob == "EnergyPlanning":
        if args.num_stages == 7:
            max_length = 80
        else:
            max_length = 90
    elif args.prob == "MertonsPortfolioOptimization":
        if args.num_stages == 7:
            max_length = 40
        else:
            max_length = 40
    elif args.prob == "ProductionPlanning":
        max_length = 100
    else:
        max_length = 100
    model.eval()
    total_loss = 0
    error_rate = []
    infeasible_cnt = 0
    infeasible_test_total_cnt = 0
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

            y_raw = tgt_raw_data[temp:temp + y_input.size(0)]
            tgt_pad_mask = model.get_pad_mask(max_seq_len=sequence_length,
                                              data=y_raw).to(device)

            temp += y_input.size(0)

            y_pred, _, _, _ = model(X, y_input, tgt_mask, tgt_pad_mask=tgt_pad_mask)

            loss = loss_fn[0](y_pred[:, :, :-4], y_answer[:, :, :-1]) + loss_fn[1](y_pred[:, :, -4:].transpose(1, 2),
                                                                                   y_answer[:, :, -1].to(torch.long))

            total_loss += loss.detach().item()

            if idx % 50 == 0:
                y_infer, _, _, _ = get_pred_cuts(X, y, 1, model, device, max_length)

                end_token_idx = get_end_token_idx(y_infer[0]) + 1
                obj_target = get_pred_obj(y[0][:, :-1].detach().cpu().data.numpy(), args)  # y_raw[0][:-1]

                obj_pred = get_pred_obj(y_infer[0][:end_token_idx, :-1].detach().cpu().data.numpy(), args)

                infeasible_test_total_cnt += 1
                if obj_pred == 0:
                    infeasible_cnt += 1
                error_rate.append(get_error_rate(obj_pred, obj_target) / np.abs(obj_target))

    return total_loss / len(dataloader), np.mean(error_rate), infeasible_cnt / infeasible_test_total_cnt


def predict(model, dataloader, args, cnt_cuts=2, device="cpu"):
    model.eval()
    pred_cut_ex, errors_pred, errors_sddp, obj_preds, obj_sddps, obj_msps = [], [], [], [], [], []

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            X, y = batch[0], batch[1]
            error_pred, error_sddp, obj_pred, obj_sddp, obj_msp, pred_cut, encoder_weights, decoder_weights_sa, decoder_weights_mha = predict_one_batch(
                X, y, idx, model, args, cnt_cuts, device)
            errors_pred += error_pred
            errors_sddp += error_sddp
            obj_preds += obj_pred
            obj_sddps += obj_sddp
            obj_msps += obj_msp
            if idx == 0:
                pred_cut_ex = pred_cut
            elif idx == 33:
                break
    return np.mean(errors_pred), np.mean(errors_sddp), np.std(errors_pred), np.std(errors_sddp), np.std(
        obj_preds), np.std(obj_sddps), np.std(obj_msps), \
           pred_cut_ex, encoder_weights, decoder_weights_sa, decoder_weights_mha


def predict_one_batch(X, y, idx, model, args, cnt_cuts=2, device="cpu"):
    X = X.to(device)
    y = y.to(device)

    if args.prob == "EnergyPlanning":
        if args.num_stages == 7:
            max_length = 80
        else:
            max_length = 90
    elif args.prob == "MertonsPortfolioOptimization":
        if args.num_stages == 7:
            max_length = 40
        else:
            max_length = 40
    elif args.prob == "ProductionPlanning":
        max_length = 100
    else:
        max_length = 100

    y_input, encoder_weights, decoder_weights_sa, decoder_weights_mha = get_pred_cuts(X, y, cnt_cuts, model, device,
                                                                                      max_length)
    if idx == 0:
        pred_cut_ex = get_all_stage_cuts(y_input, args)
    else:
        pred_cut_ex = []

    errors_pred, errors_sddp, obj_preds, obj_sddps, obj_msps = [], [], [], [], []

    for d in range(0, X.shape[0], args.num_stages - 1):
        x_curr = X[d].detach().cpu().data.numpy()
        if args.prob == "EnergyPlanning":
            paramdict = {'mean': x_curr[0, 16], 'scale': x_curr[0, 17]}
            _, optVal = MSP_EP(stageNum=args.num_stages, scenario_node=3,
                               paramdict=paramdict, mm=True)
        elif args.prob == "MertonsPortfolioOptimization":
            sigma = x_curr[0, 12] * np.sqrt(args.num_stages - 1)
            mu = (args.num_stages - 1) * x_curr[0, 8] + sigma ** 2 / 2
            paramdict = {'mu': mu, 'sigma': sigma}
            _, optVal = MSP_FP(stageNum=args.num_stages, scenario_node=3, mm=True, paramdict=paramdict)
        elif args.prob == "ProductionPlanning":
            paramdict = {'mu': list(x_curr[:3, 36]), 'sigma': list(x_curr[:3, 37])}
            _, optVal = MSP_PO(stageNum=args.num_stages, scenario_node=2, mm=True, paramdict=paramdict)
        else:
            raise NotImplementedError

        end_token_idx = get_end_token_idx(y_input[d]) + 1
        obj_target = get_pred_obj(y[d][:, :-1].detach().cpu().data.numpy(), args)  # y_raw[0][:-1]

        obj_pred = get_pred_obj(y_input[d][:end_token_idx, : -1].detach().cpu().data.numpy(), args)
        errors_pred.append(get_error_rate(obj_pred, optVal) / np.abs(optVal))
        errors_sddp.append(get_error_rate(obj_target, optVal) / np.abs(optVal))
        obj_preds.append(obj_pred)
        obj_sddps.append(obj_target)
        obj_msps.append(optVal)

        save_obj_data(data={(args.prob, args.num_stages): {'paramdict': [paramdict],
                                                           'MSP': [optVal],
                                                           'SDDP': [obj_target],
                                                           args.model: [obj_pred]}},
                      args=args)

    return errors_pred, errors_sddp, obj_preds, obj_sddps, obj_msps, pred_cut_ex, [], \
           decoder_weights_sa[end_token_idx - 2][-1][0], decoder_weights_mha[end_token_idx - 2][-1][
               0]


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

    A = np.concatenate((A, np.zeros((A.shape[0], 1))), axis=1)
    x = cp.Variable(shape=A.shape[1])

    A = np.concatenate((A, cuts[:, :-1]), axis=0)  # -1
    b = np.concatenate((b, cuts[:, -1].reshape(cuts.shape[0], 1)), axis=0)
    c = np.concatenate((c, np.array([1])), axis=0)

    constraints = [A[:num_Equality] @ x + np.squeeze(b[:num_Equality]) == 0] + [
        A[num_Equality:] @ x + np.squeeze(b[num_Equality:]) <= 0]
    if args.prob == "EnergyPlanning":
        constraints.append(x[:4] >= 0)
        obj = cp.Minimize(cp.exp(c[2] * x[1] + c[3]) + c[0] * x[2] + c[1] * x[3] + c[-1] * x[-1])
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver=cp.MOSEK)
        except:
            return 0
    elif args.prob == "MertonsPortfolioOptimization":
        constraints.append(x[:4] >= 0)
        if utility_risk_aversion_coeff == 1:
            obj = cp.Minimize(-BigM * cp.log(x[2] + smallm) + x[4])
        else:
            obj = cp.Minimize(
                -1 * (1 / (1 - utility_risk_aversion_coeff) * cp.power(x[2], 1 - utility_risk_aversion_coeff)) + x[4])
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver=cp.MOSEK)
        except:
            return 0
    else:
        constraints.append(x[:-1] >= 0)
        obj = cp.Minimize(c @ x)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.CPLEX)
    if prob.value == np.inf:
        return 0
    else:
        return prob.value


def get_error_rate(pred, target):
    return np.abs(pred - target)


def get_end_token_idx(cuts_pred):
    token_idx = cuts_pred[:, -1]
    try:
        end_token_idx = (token_idx == 3).nonzero(as_tuple=False)[0][0]
    except:
        end_token_idx = torch.tensor(cuts_pred.shape[0] - 1)

    return end_token_idx


def get_all_stage_cuts(y, args):
    cuts = []
    for i in range(args.num_stages - 1):
        cuts.append(y[i][:get_end_token_idx(y[i]) + 1].detach().cpu().data.numpy())
    return cuts


def get_pred_cuts(X, y, cnt_cuts, model, device, max_length=100):
    y_input = y[:, :cnt_cuts]
    encoder_weights = []
    decoder_weights_sa = []
    decoder_weights_mha = []
    with torch.no_grad():
        for _ in range(max_length):
            # Get source mask
            tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)

            y_pred, encoder_weight, decoder_weight_sa, decoder_weight_mha = model(X, y_input, tgt_mask)
            encoder_weights.append(encoder_weight)
            decoder_weights_sa.append(decoder_weight_sa)
            decoder_weights_mha.append(decoder_weight_mha)
            logit_to_label = torch.argmax(y_pred[:, -1, -4:], dim=1, keepdim=True)
            next_item = torch.unsqueeze(torch.concat((y_pred[:, -1, :-4], logit_to_label), dim=1), 1)

            # concatenate previous input with predicted cut
            y_input = torch.concat((y_input, next_item), dim=1)

    return y_input, encoder_weights, decoder_weights_sa, decoder_weights_mha


def save_obj_data(data, args):
    new_data = {}
    list_path = ['D:/sddp_data', "obj_data"]
    save_path = os.path.join(*list_path)
    os.makedirs(save_path, exist_ok=True)

    if not os.path.exists(os.path.join(save_path, f"{args.prob}_{args.num_stages}_{args.model}.pickle")):
        with open(os.path.join(save_path, f"{args.prob}_{args.num_stages}_{args.model}.pickle"), "wb") as fw:
            pickle.dump({(args.prob, args.num_stages): {
                "paramdict": [],
                "MSP": [],
                "SDDP": [],
                "transformer": [],
                "transformer_decoder": [],
                "L1": [],
                "VFGL": [],
                "neural_SDDP": [],
            }}, fw)

    with open(os.path.join(save_path, f"{args.prob}_{args.num_stages}_{args.model}.pickle"), "rb") as fr:
        prev_data = pickle.load(fr)

    for key, value in data.items():
        prev_data[key]["paramdict"] += value["paramdict"]
        prev_data[key]["MSP"] += value["MSP"]
        prev_data[key]["SDDP"] += value["SDDP"]
        prev_data[key][args.model] += value[args.model]
        new_data[key] = prev_data[key]

    with open(os.path.join(save_path, f"{args.prob}_{args.num_stages}_{args.model}.pickle"), "wb") as fw:
        pickle.dump(new_data, fw)
