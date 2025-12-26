import os
import numpy as np
import time
import datetime
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score,roc_auc_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, brier_score_loss
from arguments import parse_arguments
import archs as models
from archs import *
import helpers as hf
from reporter import Reporter
from datapreprocesing import preprocess_data, create_dataloaders, inverse_transform, create_binary_labels



def setup_environment(args):
    hf.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Running on {device}")
    return device


def detect_architecture(experiment_path):
    report_files = [os.path.join(experiment_path, f) for f in os.listdir(experiment_path) if f.endswith("_report.json")]

    for rp in report_files:
        arc_name = hf.extract_arc_from_report(rp)
        if arc_name:
            print(f"Detected architecture: {arc_name}")
            return arc_name
        
    raise RuntimeError("Could not detect architecture from report files")


def find_model_weights(experiment_path):
    model_files = [f for f in os.listdir(experiment_path) if f.endswith(".pt") and "entire" not in f]
    if not model_files:
        raise FileNotFoundError("No model checkpoint found")

    return os.path.join(experiment_path, model_files[0])


def prepare_test_data(args):
    threshold = 200 if "RG1" in args.file_path else 300

    data = preprocess_data(args.file_path,lookback=args.lookback,forecast=args.forecast,threshold=threshold,numfeats=args.num_feats,testonly=True)
    (X_train, y_train, _, _, _), (X_val, y_val, _, _, _), (X_test, y_test, _, _, _), scaler, scaler_y, train_std = data
    _, _, test_loader = create_dataloaders(X_train, y_train,X_val, y_val,X_test, y_test,scaler=scaler,train_std=train_std,batch_size=args.batch_size,)

    return test_loader, scaler, scaler_y


def load_model(args, arc_name, state_dict_path, X_sample, device):
    model_args = {
        "input_size": X_sample.shape[-1] if X_sample.ndim > 2 else 1,
        "hidden_size": args.hidden_units,
        "num_layers": args.num_layers,
        "forecast_len": args.forecast,
        "lookback": args.lookback,
        "dropout": args.dropout,
    }

    model = models.SerializableModule().create(arc_name, **model_args)

    state_dict = torch.load(state_dict_path, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    model.to(device)
    model.eval()

    return model


def run_inference(model, test_loader, scaler, device):
    preds_reg, preds_cl = [], []
    targets_reg, targets_cl = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.unsqueeze(-1).to(device)
            y_batch = y_batch.to(device)

            out_reg, out_cl = model(X_batch)

            preds_reg.append(out_reg.cpu())
            preds_cl.append(out_cl.cpu())
            targets_reg.append(y_batch.cpu())

            y_inv = inverse_transform(y_batch.squeeze(1).cpu().numpy(), scaler)
            
            y_bin = create_binary_labels(y_inv)
            targets_cl.append(torch.tensor(y_bin))

    return (torch.cat(preds_reg).squeeze().numpy(),torch.sigmoid(torch.cat(preds_cl).squeeze()).numpy(),
            torch.cat(targets_reg).squeeze().numpy(),torch.cat(targets_cl).squeeze().numpy())


def compute_metrics(y_true, y_true_bin, y_pred_reg, y_pred_cl, args):
    y_pred_bin = (y_pred_cl > 0.5).astype(int)

    acc = accuracy_score(y_true_bin, y_pred_bin)
    ci_acc = hf.bootstrap_ci(y_true_bin, y_pred_bin, accuracy_score)

    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    ci_prec = hf.bootstrap_ci(y_true_bin, y_pred_bin,lambda y, t: precision_score(y, t, zero_division=0))

    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    ci_rec = hf.bootstrap_ci(y_true_bin, y_pred_bin,lambda y, t: recall_score(y, t, zero_division=0),)

    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    ci_f1 = hf.bootstrap_ci(y_true_bin, y_pred_bin,lambda y, t: f1_score(y, t, zero_division=0),)

    auc = roc_auc_score(y_true_bin, y_pred_cl)
    ci_auc = hf.bootstrap_ci(y_true_bin, y_pred_cl, roc_auc_score)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
    fpr = fp / (fp + tn)
    ci_fpr = hf.bootstrap_ci(y_true_bin, y_pred_bin,lambda y, t: confusion_matrix(y, t).ravel()[1] (confusion_matrix(y, t).ravel()[0] +confusion_matrix(y, t).ravel()[1]),)

    brier = brier_score_loss(y_true_bin, y_pred_cl)
    ci_brier = hf.bootstrap_ci(y_true_bin, y_pred_cl, brier_score_loss)

    if "R2C" in args.arc:
        mae = mean_absolute_error(y_true, y_pred_reg)
        ci_mae = hf.bootstrap_ci(y_true, y_pred_reg, mean_absolute_error)

        mse = mean_squared_error(y_true, y_pred_reg)
        ci_mse = hf.bootstrap_ci(y_true, y_pred_reg, mean_squared_error)

        r2 = r2_score(y_true, y_pred_reg)
        ci_r2 = hf.bootstrap_ci(y_true, y_pred_reg, r2_score)
    else:
        mae = ci_mae = mse = ci_mse = r2 = ci_r2 = np.nan

    return {
        "accuracy": acc, "accuracy_95ci": ci_acc,
        "precision": prec, "precision_95ci": ci_prec,
        "recall": rec, "recall_95ci": ci_rec,
        "f1_score": f1, "f1_95ci": ci_f1,
        "roc_auc": auc, "roc_auc_95ci": ci_auc,
        "false_positive_rate": fpr, "fpr_95ci": ci_fpr,
        "brier_score": brier, "brier_95ci": ci_brier,
        "mae": mae, "mae_95ci": ci_mae,
        "mse": mse, "mse_95ci": ci_mse,
        "r2_score": r2, "r2_95ci": ci_r2,
    }



def main():
    args = parse_arguments()
    experiment_path = args.checkpoint
    device = setup_environment(args)

    arc_name = detect_architecture(experiment_path)
    state_dict_path = find_model_weights(experiment_path)

    test_loader, scaler, scaler_y = prepare_test_data(args)

    X_sample, _ = next(iter(test_loader))
    model = load_model(args, arc_name, state_dict_path,X_sample, device)

    print("\nOnly testing mode...")

    y_pred_reg, y_pred_cl, y_true, y_true_bin = run_inference(model, test_loader, scaler, device)

    metrics = compute_metrics(y_true, y_true_bin, y_pred_reg, y_pred_cl, args)
    soft_metrics = hf.soft_metrics(y_true_bin, y_pred_cl, figure_path=None)

    folder_name = os.path.basename(experiment_path)
    
    reporter = Reporter(experiment_path,folder_name + "_metrics_with_soft_report.json",)
    reporter.report("test_metrics", metrics)
    reporter.report("soft_metrics", soft_metrics)

    print("\nTest evaluation complete.")


if __name__ == "__main__":
    main()
