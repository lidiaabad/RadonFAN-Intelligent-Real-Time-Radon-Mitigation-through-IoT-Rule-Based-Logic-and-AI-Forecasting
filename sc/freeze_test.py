import os
import numpy as np
import time
import datetime
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score,roc_auc_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, brier_score_loss
from sklearn.metrics import ConfusionMatrixDisplay
from torch.optim.lr_scheduler import ReduceLROnPlateau
from arguments import parse_arguments
import archs as models
from archs import *
import helpers as hf
from reporter import Reporter
from datapreprocesing import preprocess_data, create_dataloaders, inverse_transform, create_binary_labels
from loss_custom import FocalLoss, DiceLoss, LogCoshLoss, QuantileLoss, MSLE, DoubleQuantileLoss, DynamicExtremeWeightedLoss, WeightedMSELoss, DenseLoss, HingeLossWithPosWeight, WeightedTemporalMSELoss, DenseBandLoss



def main():
    args = parse_arguments()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    input_file = args.file_path.replace(".csv", "")
    experiment_path= args.checkpoint
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Running on {device}")

    #Find the report file to load arc name
    report_files = [os.path.join(experiment_path, f) for f in os.listdir(experiment_path)if f.endswith("_report.json")]
    arc_name = None
    for rp in report_files:
        arc_name = hf.extract_arc_from_report(rp)
        if arc_name:
            break
    print(f"Detected architecture: {arc_name}")
    
    # Load model
    model_files = [f for f in os.listdir(experiment_path) if f.endswith(".pt") and "entire" not in f]
    state_dict_path = os.path.join(experiment_path, model_files[0])
    model_args = {
        'input_size': 1,
        'hidden_size': args.hidden_units,
        'num_layers': args.num_layers,
        'forecast': args.forecast,
    }
    model = models.SerializableModule().create(arc_name, **model_args)
    state_dict = torch.load(state_dict_path, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    model.to(device)

    # New report name only for metrics
    folder_name = os.path.basename(experiment_path)  
    reporter = Reporter(experiment_path, folder_name + '_metrics_with_soft_report.json')

    hf.set_seed(args.seed)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, train_std = preprocess_data(args.file_path)
    _, _, test_loader = create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                                                               scaler=scaler, train_std=train_std, batch_size=args.batch_size)

    print("\nOnly testing mode...")
    model.eval()
    test_preds_reg, test_preds_cl, test_targets_reg, test_targets_cl = [], [], [], []

    with torch.no_grad():
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch = X_test_batch.unsqueeze(-1).to(device)
            outputs_reg, outputs_cl = model(X_test_batch)
            test_preds_reg.append(outputs_reg.cpu())
            test_preds_cl.append(outputs_cl.cpu())
            test_targets_reg.append(y_test_batch.cpu())

            y_batch_inv = inverse_transform(y_test_batch.squeeze(1).detach().cpu().numpy(), scaler)
            y_batch_bin = create_binary_labels(y_batch_inv)
            test_targets_cl.append(torch.tensor(y_batch_bin))

    test_preds_reg = torch.cat(test_preds_reg).squeeze()
    test_preds_cl = torch.cat(test_preds_cl).squeeze()
    test_targets_reg = torch.cat(test_targets_reg).squeeze()
    test_targets_cl = torch.cat(test_targets_cl).squeeze()

    y_pred_reg = test_preds_reg.numpy()
    y_pred_cl = torch.sigmoid(test_preds_cl).numpy()
    y_pred_bin = (y_pred_cl > 0.5).astype(int) #0.5
    y_true = test_targets_reg.numpy()
    y_true_bin = test_targets_cl.numpy()

    # Hard metrics with bootstrap CI
    acc = accuracy_score(y_true_bin, y_pred_bin)
    ci_acc = hf.bootstrap_ci(y_true_bin, y_pred_bin, accuracy_score)

    prec = precision_score(y_true_bin, y_pred_bin)
    ci_prec = hf.bootstrap_ci(y_true_bin, y_pred_bin, lambda y,t: precision_score(y,t,zero_division=0))

    rec = recall_score(y_true_bin, y_pred_bin)
    ci_rec = hf.bootstrap_ci(y_true_bin, y_pred_bin, lambda y,t: recall_score(y,t,zero_division=0))

    f1 = f1_score(y_true_bin, y_pred_bin)
    ci_f1 = hf.bootstrap_ci(y_true_bin, y_pred_bin, lambda y,t: f1_score(y,t,zero_division=0))

    auc = roc_auc_score(y_true_bin, y_pred_cl)
    ci_auc = hf.bootstrap_ci(y_true_bin, y_pred_cl, roc_auc_score)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
    fpr = fp / (fp + tn)
    ci_fpr = hf.bootstrap_ci(y_true_bin, y_pred_bin, lambda y,t: confusion_matrix(y,t).ravel()[1]/(confusion_matrix(y,t).ravel()[0]+confusion_matrix(y,t).ravel()[1]))

    brier_score_val = brier_score_loss(y_true_bin, y_pred_cl)
    ci_brier = hf.bootstrap_ci(y_true_bin, y_pred_cl, brier_score_loss)

    # Regression metrics with bootstrap CI
    mae = mean_absolute_error(y_true, y_pred_reg)
    ci_mae = hf.bootstrap_ci(y_true, y_pred_reg, mean_absolute_error)

    mse = mean_squared_error(y_true, y_pred_reg)
    ci_mse = hf.bootstrap_ci(y_true, y_pred_reg, mean_squared_error)

    r2 = r2_score(y_true, y_pred_reg)
    ci_r2 = hf.bootstrap_ci(y_true, y_pred_reg, r2_score)

    test_metrics = {
        'accuracy': acc, 'accuracy_95ci': ci_acc,
        'precision': prec, 'precision_95ci': ci_prec,
        'recall': rec, 'recall_95ci': ci_rec,
        'f1_score': f1, 'f1_95ci': ci_f1,
        'roc_auc': auc, 'roc_auc_95ci': ci_auc,
        'false_positive_rate': fpr, 'fpr_95ci': ci_fpr,
        'brier_score': brier_score_val, 'brier_95ci': ci_brier,
        'mae': mae, 'mae_95ci': ci_mae,
        'mse': mse, 'mse_95ci': ci_mse,
        'r2_score': r2, 'r2_95ci': ci_r2
    }

    soft_metrics = hf.soft_metrics(y_true_bin, y_pred_cl, figure_path=None)

    # Report metrics
    reporter.report('test_metrics', test_metrics)
    reporter.report('soft_metrics', soft_metrics)
    print("\nTest evaluation complete.")

if __name__ == "__main__":
    main()