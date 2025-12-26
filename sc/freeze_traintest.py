import os
import numpy as np
import time
import datetime
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, brier_score_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from arguments import parse_arguments
import archs as models
import helpers as hf
from reporter import Reporter
from datapreprocesing import preprocess_data, create_dataloaders
from loss_custom import DenseLoss,compute_loss_per_class
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def check_path(folder_name):
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
        
        
def setup_environment(args):
    hf.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu" )
    print(f"Running on {device}")
    return device


def setup_experiment(args):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    input_file = args.file_path.replace(".csv", "")
    experiment_path = os.path.join( args.checkpoint, f"radon_{input_file}{args.arc}{args.model_id}")

    check_path(args.checkpoint)
    check_path(experiment_path)

    reporter = Reporter(experiment_path, f"{args.model_id}_report.json")
    reporter.report("arguments", vars(args))
    reporter.report("experiment_date", current_time)

    return experiment_path, reporter


def prepare_data(args):
    threshold = 200 if "RG1" in args.file_path else 300

    data = preprocess_data(args.file_path,lookback=args.lookback,forecast=args.forecast,threshold=threshold,numfeats=args.num_feats)
    loaders = create_dataloaders(*data[:3],scaler=data[3],train_std=data[4],batch_size=args.batch_size,)

    return { "raw": data,"loaders": loaders,"threshold": threshold, "scaler_y": data[3]}


def build_model(args, data, device, reporter):
    X_train = data["raw"][0][0]

    model_args = {
        "input_size": X_train.shape[-1] if X_train.ndim > 2 else 1,
        "hidden_size": args.hidden_units,
        "num_layers": args.num_layers,
        "forecast_len": args.forecast,
        "lookback": args.lookback,
        "dropout": args.dropout,
    }

    model = models.SerializableModule().create(args.arc, **model_args)
    model.apply(init_weights)
    model.to(device)

    sample_batch = next(iter(data["loaders"][0]))[0].to(device)
    cc = hf.get_computational_cost(model, (sample_batch,))
    reporter.report("computational_cost", cc)

    print(model)
    return model


def init_weights(m):
    if isinstance(m, nn.LSTM):
        # Initialize LSTM weights using Xavier initialization
        torch.nn.init.xavier_uniform_(m.weight_ih_l0)  # Input-to-hidden weights
        torch.nn.init.xavier_uniform_(m.weight_hh_l0)  # Hidden-to-hidden weights
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias_ih_l0)  # Initialize input-to-hidden biases to zero
            torch.nn.init.zeros_(m.bias_hh_l0)  # Initialize hidden-to-hidden biases to zero
    
    if isinstance(m, nn.Linear):
        # Initialize Linear layer weights using He initialization
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # For ReLU activations
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            
    if isinstance(m, nn.Conv1d):
        # Initialize Conv layer weights using He initialization
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # For ReLU activations
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            
            
def freeze_regression(model):
    """Freeze classification layers during regression training."""
    for name, module in model.named_children():
        if "cl" in name:
            for param in module.parameters():
                param.requires_grad = False
        else: 
            for param in module.parameters():
                param.requires_grad = True
            
            
def freeze_classification(model):
    """Freeze regression-related layers during classification training."""
    for name, module in model.named_children():
        if "cl" in name:
            for param in module.parameters():
                param.requires_grad = True
        else: 
            for param in module.parameters():
                param.requires_grad = False
            

def get_phases(args):
    if "DC" in args.arc:
        return ["classification"]
    if "R2C" in args.arc:
        return ["regression", "classification"]
    raise ValueError("Unknown architecture")


def compute_loss(phase, y_reg, y_cl, y, y_bin, w_reg, w_cl,criterion_reg, criterion_cl):
    if phase == "regression":
        loss = criterion_reg(y_reg, y)
        return (w_reg.unsqueeze(1) * loss).mean()

    loss = criterion_cl(y_cl, y_bin) * w_cl
    return loss.mean()


def train_epoch(model, loader, optimizer, phase, device,criterion_reg, criterion_cl):
    
    model.train()

    loss_epoch = 0
    pos_loss_sum = neg_loss_sum = 0
    num_pos = num_neg = 0
    neg_probs = []

    for X, y, y_bin, w_cl, w_reg in loader:
        X = X.to(device)
        y = y.squeeze(-1).to(device)
        y_bin = y_bin.unsqueeze(-1).to(device)
        w_cl, w_reg = w_cl.to(device), w_reg.to(device)

        optimizer.zero_grad()
        y_reg, y_cl = model(X)

        if phase == "regression":
            loss_reg = criterion_reg(y_reg, y)
            loss = (w_reg.unsqueeze(1) * loss_reg).mean()

        else:
            loss_cl = criterion_cl(y_cl, y_bin) * w_reg
            loss = loss_cl.mean()

            pos_l, neg_l = compute_loss_per_class(y_cl, y_bin)
            p = (y_bin == 1).sum().item()
            n = (y_bin == 0).sum().item()

            pos_loss_sum += pos_l * p
            neg_loss_sum += neg_l * n
            num_pos += p
            num_neg += n

            probs = torch.sigmoid(y_cl).detach().cpu()
            neg_probs.extend(probs[y_bin.cpu() == 0].numpy())

        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()

    return {"loss": loss_epoch / len(loader),"pos_loss": pos_loss_sum / max(num_pos, 1),"neg_loss": neg_loss_sum / max(num_neg, 1),"neg_probs": neg_probs}

def val_epoch(model, loader, phase, device, criterion_reg, criterion_cl):
    
    model.eval()

    loss_epoch = 0
    pos_loss_sum = neg_loss_sum = 0
    num_pos = num_neg = 0
    neg_probs = []

    with torch.no_grad():
        for X, y, y_bin, _, _ in loader:
            X = X.to(device)
            y = y.squeeze(-1).to(device)
            y_bin = y_bin.unsqueeze(-1).to(device)

            y_reg, y_cl = model(X)

            if phase == "regression":
                loss = criterion_reg(y_reg, y).mean()
            else:
                loss = criterion_cl(y_cl, y_bin).mean()

                pos_l, neg_l = compute_loss_per_class(y_cl, y_bin)
                p = (y_bin == 1).sum().item()
                n = (y_bin == 0).sum().item()

                pos_loss_sum += pos_l * p
                neg_loss_sum += neg_l * n
                num_pos += p
                num_neg += n

                probs = torch.sigmoid(y_cl).cpu()
                neg_probs.extend(probs[y_bin.cpu() == 0].numpy())

            loss_epoch += loss.item()

    return {"loss": loss_epoch / len(loader),"pos_loss": pos_loss_sum / max(num_pos, 1),"neg_loss": neg_loss_sum / max(num_neg, 1),"neg_probs": neg_probs}


def train_phase(model, phase, args, loaders,device, reporter, experiment_path):
    print(f"\nStarting {phase.upper()} phase")

    if phase == "regression":
        freeze_regression(model)
        lr = args.lr / 10
    else:
        freeze_classification(model)
        lr = args.lr

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=lr,weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, "min", patience=10)
    criterion_reg = DenseLoss()
    criterion_cl = nn.BCEWithLogitsLoss(reduction="none")

    train_loader, val_loader, _ = loaders

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_pos_loss": [],
        "train_neg_loss": [],
        "val_pos_loss": [],
        "val_neg_loss": [],
    }

    best = np.inf
    patience = 0

    for epoch in range(1, args.epochs + 1):
        
        tr = train_epoch(model, train_loader, optimizer,phase, device, criterion_reg, criterion_cl)
        vl = val_epoch(model, val_loader,phase, device, criterion_reg, criterion_cl)

        scheduler.step(vl["loss"])

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(vl["loss"])
        history["train_pos_loss"].append(tr["pos_loss"])
        history["train_neg_loss"].append(tr["neg_loss"])
        history["val_pos_loss"].append(vl["pos_loss"])
        history["val_neg_loss"].append(vl["neg_loss"])

        print(
            f"Epoch {epoch}: "
            f"Train {tr['loss']:.4f} "
            f"(Pos {tr['pos_loss']:.4f}, Neg {tr['neg_loss']:.4f}) | "
            f"Val {vl['loss']:.4f} "
            f"(Pos {vl['pos_loss']:.4f}, Neg {vl['neg_loss']:.4f})"
        )

        if vl["loss"] < best:
            best = vl["loss"]
            patience = 0
            model.save_entire_model(os.path.join(experiment_path, args.model_id))
        else:
            patience += 1

        if patience >= args.patience:
            print("Early stopping")
            break

    reporter.report(f"{phase}_metrics", history)
    hf.save_train_metrics_plot(history,os.path.join(experiment_path, f"{phase}_metrics.png"),phase=phase)
    
def evaluate (model, args, loaders, device,threshold, scaler_y,experiment_path, reporter, phases): 
    
    print("\nTesting...")
    model.eval()
    _, _, test_loader = loaders
    
    test_preds_reg, test_preds_cl, test_targets_reg, test_targets_cl = [], [], [], []
    with torch.no_grad():
        for i, (X_test_batch, y_test_batch, y_test_batch_bin, w_cl_batch, w_reg_batch) in enumerate(test_loader):
            X_test_batch = X_test_batch.to(device)
            y_test_batch = y_test_batch.to(device)
            y_test_batch_bin = y_test_batch_bin.to(device)

            if getattr(args, "ar", 1) > 1:
                '''AR modelo: train model to t+1 and apply it fc times'''
                B,_, _ = X_test_batch.shape
                ar_horizon = getattr(args, "ar", 1)
                if i + ar_horizon - 1 >= len(test_loader.dataset):
                    break

                X_ar = X_test_batch.clone()
                preds_ar = torch.zeros(B, ar_horizon).to(device)
                true_ar = torch.zeros(B, ar_horizon).to(device)

                for step in range(ar_horizon):
                    outputs_reg, outputs_cl = model(X_ar)
                    preds_ar[:, step] = outputs_reg.view(-1)
                    true_future = test_loader.dataset[i + step][1]  
                    true_ar[:, step] = torch.tensor(true_future, device=device).view(-1)
                    new_val = outputs_reg.detach().unsqueeze(1)  # (B,1,1)
                    X_ar = torch.cat([X_ar[:, 1:, :], new_val], dim=1)
                    
                test_preds_reg.append(preds_ar.cpu())
                test_targets_reg.append(true_ar.cpu())
                last_cl = test_loader.dataset[i + ar_horizon - 1][2]  
                test_preds_cl.append(outputs_cl.cpu())
                test_targets_cl.append(torch.tensor(last_cl.unsqueeze(0)).cpu())
            
            else:
                '''Direct model'''
                outputs_reg, outputs_cl = model(X_test_batch)
                test_preds_reg.append(outputs_reg.cpu())
                test_targets_reg.append(y_test_batch.cpu())
                test_preds_cl.append(outputs_cl.cpu())
                test_targets_cl.append(y_test_batch_bin.cpu())

    # Concatenate all
    if args.ar > 1: 
        test_preds_reg = torch.cat(test_preds_reg)
        test_targets_reg = torch.cat(test_targets_reg)
    else: 
        test_preds_reg = torch.cat(test_preds_reg).squeeze()
        test_targets_reg = torch.cat(test_targets_reg).squeeze()
    
    test_preds_cl = torch.cat(test_preds_cl).squeeze()
    test_targets_cl = torch.cat(test_targets_cl).squeeze()

    y_pred_reg = test_preds_reg.numpy()
    y_pred_cl = torch.sigmoid(test_preds_cl).numpy()
    y_true = test_targets_reg.numpy()
    y_true_bin = test_targets_cl.numpy()    
    
    mode, thr_up, thr_down, trend, slope=  hf.grid_search_smoothing(y_true_bin, y_pred_cl)
    y_pred_bin= (y_pred_cl > 0.5).astype(int)

    if 'regression' in phases:
        figure_path_reg = os.path.join(experiment_path, 'test_pred_reg.png')
        hf.plot_y_pred_vs_target(y_true, y_pred_reg, y_pred_bin, y_true, scaler_y, threshold, figure_path=figure_path_reg, task='regression')
    
    figure_path_cl = os.path.join(experiment_path, 'test_pred_cl.png')
    hf.plot_y_pred_vs_target(y_true, y_pred_cl, y_pred_bin, y_true_bin, scaler_y, threshold, figure_path=figure_path_cl, task='classification')

    figure_path_pos = os.path.join(experiment_path, 'positive_indices.png')
    hf.plot_positive_indices(y_true_bin, y_pred_bin, figure_path=figure_path_pos)

    acc = accuracy_score(y_true_bin, y_pred_bin)
    ci_acc = hf.bootstrap_ci(y_true_bin, y_pred_bin, accuracy_score)
    prec = precision_score(y_true_bin, y_pred_bin)
    ci_prec = hf.bootstrap_ci(y_true_bin, y_pred_bin, lambda y,t: precision_score(y,t,zero_division=0))
    rec = recall_score(y_true_bin, y_pred_bin)
    ci_rec = hf.bootstrap_ci(y_true_bin, y_pred_bin, lambda y,t: recall_score(y,t,zero_division=0))
    f1 = f1_score(y_true_bin, y_pred_bin)
    ci_f1 = hf.bootstrap_ci(y_true_bin, y_pred_bin, lambda y,t: f1_score(y,t,zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
    fpr = fp / (fp + tn)
    ci_fpr = hf.bootstrap_ci(y_true_bin, y_pred_bin, lambda y,t: confusion_matrix(y,t).ravel()[1]/(confusion_matrix(y,t).ravel()[0]+confusion_matrix(y,t).ravel()[1]))
    brier_score_val = brier_score_loss(y_true_bin, y_pred_cl)
    ci_brier = hf.bootstrap_ci(y_true_bin, y_pred_cl, brier_score_loss)
    
    if 'regression' in phases:
        mae = mean_absolute_error(y_true, y_pred_reg)
        ci_mae = hf.bootstrap_ci(y_true, y_pred_reg, mean_absolute_error)
        mse = mean_squared_error(y_true, y_pred_reg)
        ci_mse = hf.bootstrap_ci(y_true, y_pred_reg, mean_squared_error)
        r2 = r2_score(y_true, y_pred_reg)
        ci_r2 = hf.bootstrap_ci(y_true, y_pred_reg, r2_score)
    else:
        mae = ci_mae = mse = ci_mse = r2 = ci_r2 = np.nan

    test_metrics = {
        'accuracy': acc, 'accuracy_95ci': ci_acc,
        'precision': prec, 'precision_95ci': ci_prec,
        'recall': rec, 'recall_95ci': ci_rec,
        'f1_score': f1, 'f1_95ci': ci_f1,
        'false_positive_rate': fpr, 'fpr_95ci': ci_fpr,
        'brier_score': brier_score_val, 'brier_95ci': ci_brier,
        'mae': mae, 'mae_95ci': ci_mae,
        'mse': mse, 'mse_95ci': ci_mse,
        'r2_score': r2, 'r2_95ci': ci_r2
    }

    figure_path_soft = os.path.join(experiment_path, 'soft_conf_matrix.png')
    soft_metrics=hf.soft_metrics(y_true_bin, y_pred_cl, figure_path=figure_path_soft, threshold=0.5)

    figure_path = os.path.join(experiment_path, 'fp_fn.png')
    if 'regression' in phases:
        hf.analyze_classification_errors(y_pred_bin, y_pred_cl, y_true_bin, y_true, y_pred_reg, scaler_y, classifier=False, figure_path=figure_path)
    else: 
        hf.analyze_classification_errors(y_pred_bin, y_pred_cl, y_true_bin, y_true, y_pred_reg, scaler_y, classifier=True, figure_path=figure_path)

    figure_path_cm = os.path.join(experiment_path, 'confusion_matrix.png')
    hf.conf_matrix(y_true_bin, y_pred_bin, figure_path=figure_path_cm)

    figure_path_cl = os.path.join(experiment_path, 'hist_cl.png')
    hf.plot_y_pred_vs_target_hist(y_pred_cl, y_true_bin, y_true, figure_path=figure_path_cl)

    reporter.report('test_metrics', test_metrics)
    reporter.report('soft_metrics', soft_metrics)
    
    figure_path= os.path.join(experiment_path, 'RSvsIA.png')
    ai_rs=hf.comparison_to_rule(args.file_path, model, device, scaler_y, threshold, lookback=args.lookback, forecast=args.forecast, thr_up=thr_up, thr_down=thr_down, trend=trend, slope_rel=slope, mode=mode, figure_path=figure_path)
    reporter.report('cost_safety',ai_rs)
    
    
def main():
    args = parse_arguments()
    device = setup_environment(args)
    experiment_path, reporter = setup_experiment(args)
    
    data, loaders, threshold = prepare_data(args)
    X_train = data[0][0]
    model = build_model(args, X_train, device, reporter)
    
    phases = phase in get_phases(args) 
    
    for phase in phases:
        train_phase(model, phase, args, loaders,device, reporter, experiment_path)

    evaluate(model, args, loaders, device,threshold, data[3],experiment_path, reporter, phases)
    print("\nTraining and evaluation complete.")


if __name__ == "__main__":
    main()
