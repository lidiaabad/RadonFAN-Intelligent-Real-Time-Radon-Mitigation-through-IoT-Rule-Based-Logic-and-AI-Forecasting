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
import helpers as hf
from reporter import Reporter
from datapreprocesing import preprocess_data, create_dataloaders, inverse_transform, create_binary_labels
from loss_custom import DenseLoss, DenseLossProb, AsymmetricMSELoss, AsymmetricPersistenceMSELoss, WeightedLagDenseLoss, compute_loss_per_class, DenseLossPersistence
import warnings
import matplotlib.pyplot as plt



warnings.filterwarnings("ignore")



def check_path(folder_name):
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

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
        # Initialize Linear layer weights using He initialization
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # For ReLU activations
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def safe_stat(t, fn):
    return fn(t).item() if t.numel() > 0 else float('nan')


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
            


features_before, features_after= [], []


def hook_before(module, inp, out):
    # inp[0] = cl_input [B, H*5]
    features_before.append(inp[0].detach().cpu())
def hook_after(module, inp, out):
    # out = salida penúltima capa [B, hidden_size//2]
    features_after.append(out.detach().cpu())

def main():
    args = parse_arguments()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_id = args.model_id
    input_file = args.file_path.replace(".csv", "")
    experiment_path = os.path.join(args.checkpoint, 'radon_' + input_file + args.arc + model_id)
    check_path(args.checkpoint)
    check_path(experiment_path)

    reporter = Reporter(experiment_path, model_id + '_report.json')
    reporter.report('arguments', vars(args))
    reporter.report('experiment_date', current_time)

    hf.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Running on {device}")
    
    #discriminator = models.Discriminator(input_dim=1, hidden_dim=64).to(device)
    #optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

    if 'RG1' in args.file_path: 
        threshold=200

    elif 'RG2' in args.file_path: 
        threshold=300
        
    (X_train, y_train, y_train_bin, w_cl_train, w_reg_train), (X_val, y_val, y_val_bin, w_cl_val, w_reg_val), (X_test, y_test, y_test_bin, w_cl_test, w_reg_test), scaler, scaler_y, train_std = preprocess_data(args.file_path, lookback=args.lookback, forecast=args.forecast, threshold=threshold, numfeats=args.num_feats)

    train_loader, val_loader, test_loader = create_dataloaders(X_train, y_train, y_train_bin, w_cl_train, w_reg_train, X_val, y_val, y_val_bin, w_cl_val, w_reg_val, X_test, y_test, y_test_bin, w_cl_test, w_reg_test,
                                                               scaler=scaler, train_std=train_std, batch_size=args.batch_size)


    #upper = torch.tensor(scaler_y.transform([[300]])[0][0], device=device)
    #lower = torch.tensor(scaler_y.transform([[200]])[0][0], device=device)

    print(X_train.shape)

    model_args = {
        'input_size': X_train.shape[-1] if X_train.ndim > 2 else 1,
        'hidden_size': args.hidden_units,
        'num_layers': args.num_layers,
        'forecast_len': args.forecast,
        'lookback': args.lookback,
        'dropout': args.dropout
    }

    model = models.SerializableModule().create(args.arc, **model_args)
    model.apply(init_weights)
    model.to(device)
    model_path = os.path.join(experiment_path, model_id)
    print(model)
    
    batch = next(iter(train_loader))
    x = batch[0].to(device)  # assuming (x, y)
    cc=hf.get_computational_cost(model, (x,))
    reporter.report(f'computational_cost', cc)
    
    # --------------- Weights computing
    print("Proportion of y=1 in train:", (y_train_bin == 1).sum()/len(y_train_bin))
    print("Proportion of y=1 in val:", (y_val_bin == 1).sum()/len(y_val_bin))
    print("Proportion of y=1 in test:", (y_test_bin == 1).sum()/len(y_test_bin))
    print("Proportion of y=1 in ALL:", ((y_train_bin == 1).sum() + (y_val_bin == 1).sum() + (y_test_bin == 1).sum())/(len(y_train_bin) + len(y_val_bin) + len(y_test_bin)))

    #classification
    pos_weight = torch.tensor(np.sqrt((y_train_bin == 0).sum()/(y_train_bin == 1).sum()), dtype=torch.float32).to(device)
    #pos_weight = torch.tensor(0.75*((y_train_bin == 0).sum()/(y_train_bin == 1).sum()), dtype=torch.float32).to(device)
    print("Positive samples weight:", pos_weight)


    criterion_cl_train = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none') #reduction='none'
    criterion_cl_val = nn.BCEWithLogitsLoss(reduction='none') #reduction='none'
    
    #criterion_reg=DenseLossPersistence(lambda_p=0.01)
    criterion_reg=DenseLoss()
    #criterion_reg = AsymmetricPersistenceMSELoss(alpha=2.0)
    criterion_disc = nn.BCEWithLogitsLoss()
    

    if "DC" in args.arc:
        phases = ['classification']
    elif "R2C" in args.arc:
        phases = ['regression', 'classification']
   
        
    for phase in phases:
        print(f"\n Starting phase: {phase.upper()}")
        # Freeze layers depending on phase
        if len(phases) > 1:
            if phase == 'regression':
                freeze_regression(model)
                lr=args.lr/10
            elif phase == 'classification':
                freeze_classification(model)
                lr=args.lr
            

        train_losses, train_losses_reg, train_losses_cl = [], [], []
        validation_losses, validation_losses_reg, validation_losses_cl = [], [], []
        epoch_times = []
        disc_losses, disc_val_losses = [], []
        best_val_loss = np.inf
        epochs_without_improvement = 0

        # Optimizer and scheduler for trainable parameters only
        optimizer = torch.optim.AdamW( filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        horizon=0
        print(f"\nStarting in horizon t+{horizon+1}")
        epoch_gradients = []

        for epoch in range(1, args.epochs + 1):
            grad_accumulator = {}
            batch_count = 0
            
            model.train()
            start_time = time.time()
            train_loss_epoch, train_loss_epoch_reg, train_loss_epoch_cl = 0, 0, 0
            train_pos_loss_sum, train_neg_loss_sum, val_pos_loss_sum, val_neg_loss_sum = 0, 0, 0, 0
            num_pos_total, num_neg_total, val_num_pos_total, val_num_neg_total = 0, 0, 0, 0
            disc_loss_epoch, disc_val_loss_epoch=0, 0
            neg_probs_train, neg_probs_val = [], []

            for X_batch, y_batch, y_batch_bin, w_cl_batch, w_reg_batch in train_loader:
                #X_batch = X_batch.unsqueeze(-1).to(device)
                X_batch = X_batch.to(device)
                if y_batch.ndim == 3: 
                   y_batch = y_batch.squeeze(-1)
                y_batch = y_batch.to(device)
                y_batch_bin = y_batch_bin.unsqueeze(-1).to(device)
                w_cl_batch = w_cl_batch.to(device)
                w_reg_batch = w_reg_batch.to(device)
                optimizer.zero_grad()
                y_pred_reg, y_pred_cl = model(X_batch)

                if phase == 'regression':

                    loss_reg = criterion_reg(y_pred_reg, y_batch)
                    loss_main = (w_reg_batch.unsqueeze(1) *loss_reg).mean()  #w_reg_batch.unsqueeze(1) *

                    loss=loss_main
                    train_loss_epoch_reg += loss.item()

                else:
                    loss_cl = criterion_cl_train(y_pred_cl, y_batch_bin) *w_reg_batch 
                    loss = loss_cl.mean()
                    train_loss_epoch_cl += loss.item()
                    pos_loss, neg_loss = compute_loss_per_class(y_pred_cl, y_batch_bin) #  pos_weight, w_reg_batch///w_cl_batch) 
                    num_pos = (y_batch_bin == 1).sum().item()
                    num_neg = (y_batch_bin == 0).sum().item()
                    probs = torch.sigmoid(y_pred_cl).detach().cpu()
                    neg_probs_train.extend(probs[y_batch_bin.detach().cpu() == 0].numpy())
                    #print(f"num pos and num negs:{num_pos}, {num_neg}")
                    train_pos_loss_sum += pos_loss * num_pos
                    train_neg_loss_sum += neg_loss * num_neg
                    num_pos_total += num_pos
                    num_neg_total += num_neg

                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # === Registro de gradientes: media por época ===

                for name, param in model.named_parameters():
                    if param.requires_grad:
                        g = param.grad
                        if g is not None:
                            norm = g.norm().item()
                            if name not in grad_accumulator:
                                grad_accumulator[name] = 0.0
                            grad_accumulator[name] += norm
                
                optimizer.step()
                batch_count += 1

                train_loss_epoch += loss.item()

            epoch_stats = {name: total/batch_count for name, total in grad_accumulator.items()}
            epoch_gradients.append(epoch_stats)
            
            # === Validación ===
            model.eval()
            val_loss_epoch, val_loss_epoch_reg, val_loss_epoch_cl = 0, 0, 0

            with torch.no_grad():
                for X_val_batch, y_val_batch, y_val_batch_bin, w_cl_batch, w_reg_batch in val_loader:
                    #X_val_batch = X_val_batch.unsqueeze(-1).to(device)
                    X_val_batch = X_val_batch.to(device)
                    if y_val_batch.ndim == 3: 
                        y_val_batch = y_val_batch.squeeze(-1)
                    y_val_batch = y_val_batch.to(device)
                    y_val_batch_bin = y_val_batch_bin.unsqueeze(-1).to(device)

                    y_val_pred_reg, y_val_pred_cl = model(X_val_batch)

                    if phase == 'regression':

                        val_loss = (criterion_reg(y_val_pred_reg, y_val_batch)).mean()
                        val_loss_epoch += val_loss.item()
                        val_loss_epoch_reg += val_loss.item()
                        
                    else:
                        val_loss_cl = criterion_cl_val(y_val_pred_cl, y_val_batch_bin).mean()
                        val_loss_epoch += val_loss_cl.item()
                        val_loss_epoch_cl += val_loss_cl.item()

                        pos_loss, neg_loss = compute_loss_per_class(y_val_pred_cl, y_val_batch_bin)
                        num_pos = (y_val_batch_bin == 1).sum().item()
                        num_neg = (y_val_batch_bin == 0).sum().item()
                        probs = torch.sigmoid(y_val_pred_cl).cpu()
                        neg_probs_val.extend(probs[y_val_batch_bin.cpu()  == 0].numpy())

                        #print(f"num pos and num negs:{num_pos}, {num_neg}")
                        val_pos_loss_sum += pos_loss * num_pos
                        val_neg_loss_sum += neg_loss * num_neg
                        val_num_pos_total += num_pos
                        val_num_neg_total += num_neg

            # === Estadísticas por época ===
            #print("Train neg mean prob:", np.mean(neg_probs_train))
            #print("Val neg mean prob:", np.mean(neg_probs_val))
            
            train_loss_epoch /= len(train_loader)
            train_loss_epoch_reg /= len(train_loader)
            train_loss_epoch_cl /= len(train_loader)
            val_loss_epoch /= len(val_loader)
            val_loss_epoch_reg /= len(val_loader)
            val_loss_epoch_cl /= len(val_loader)

            if phase == 'classification':
                train_pos_loss = train_pos_loss_sum / num_pos_total
                train_neg_loss = train_neg_loss_sum / num_neg_total
                val_pos_loss = val_pos_loss_sum / val_num_pos_total
                val_neg_loss = val_neg_loss_sum / val_num_neg_total

            train_losses.append(train_loss_epoch)
            train_losses_reg.append(train_loss_epoch_reg)
            train_losses_cl.append(train_loss_epoch_cl)
            validation_losses.append(val_loss_epoch)
            validation_losses_reg.append(val_loss_epoch_reg)
            validation_losses_cl.append(val_loss_epoch_cl)
            


            scheduler.step(val_loss_epoch)

            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                epochs_without_improvement = 0
                model.save_entire_model(model_path)
                model.save(model_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement % 5 == 0:
                    print(f'\t Validation loss did not improve {val_loss_epoch :.4f} vs {best_val_loss :.4f}. Patience {epochs_without_improvement}/{args.patience}.')
            
            if epochs_without_improvement >= args.patience:
                '''
                if phase == 'regression' and horizon < (args.forecast - 1):
                    # Cambiamos al siguiente horizonte
                    horizon += 1
                    print(f"Weights: {weights_r}")
                    print(f"\nSwitching to horizon t+{horizon+1}")
                    
                    # Reiniciamos optimizer, scheduler y early stopping
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=args.lr,
                        weight_decay=args.weight_decay
                    )
                    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
                    best_val_loss = float('inf')
                    epochs_without_improvement = 0
                else:
                '''
                print(f"Early stopping triggered at final horizon t+{horizon+1} after {epoch} epochs.")
                break

            current_time = time.time()
            epoch_times.append(current_time - start_time)

            if (epoch + 1) % args.epochsinfo == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss_epoch:.4f}, Val Loss: {val_loss_epoch:.4f}, Duration: {current_time - start_time:.2f}s")
                if phase == 'classification':
                    print(f"Epoch {epoch+1}/{args.epochs}, "
                            f" Unweighted Train Loss: {train_loss_epoch:.4f} "
                            f"(Pos: {train_pos_loss:.4f}, Neg: {train_neg_loss:.4f}), "
                            f"Val Loss: {val_loss_epoch:.4f} "
                            f"(Pos: {val_pos_loss:.4f}, Neg: {val_neg_loss:.4f}), "
                            f"Duration: {current_time - start_time:.2f}s")

                for param_group in optimizer.param_groups:
                    print(f"Learning rate: {param_group['lr']:.8f}")
               
                        
        # Final metrics and plots
    
        # Graficar evolución de la media de gradientes
        colors = plt.get_cmap('tab20').colors

        # Graficar
        plt.figure(figsize=(12, 6))
        names = epoch_gradients[0].keys()

        for i, name in enumerate(names):
            color = colors[i % len(colors)]  # Cicla si hay más de 20
            values = [e[name] for e in epoch_gradients]
            plt.plot(values, label=name, color=color)

        plt.xlabel("Epoch")
        plt.ylabel("Gradient norm")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Guardar figura
        os.makedirs(experiment_path, exist_ok=True)
        plt.savefig(os.path.join(experiment_path, "gradients.png"))
        plt.close()



        metrics = {
            'training_time_s': sum(epoch_times),
            'number_of_epochs': len(epoch_times),
            'train_loss': train_losses,
            'validation_loss': validation_losses,
            'train_loss_reg': train_losses_reg,
            'validation_loss_reg': validation_losses_reg,
            'train_loss_cl': train_losses_cl,
            'validation_loss_cl': validation_losses_cl,
        }

        reporter.report(f'{phase}_metrics', metrics)
        figure_path = os.path.join(experiment_path, f'{phase}_metrics.png')
        hf.save_train_metrics_plot(metrics, figure_path, phase=phase)
        #figure_path = os.path.join(experiment_path, 'discriminator_loss.png')
        #hf.plot_discriminator_loss(disc_losses, disc_val_losses, figure_path)

    print("\nTesting...")
    model.eval()
    test_preds_reg, test_preds_cl, test_targets_reg, test_targets_cl = [], [], [], []
    with torch.no_grad():
        for i, (X_test_batch, y_test_batch, y_test_batch_bin, w_cl_batch, w_reg_batch) in enumerate(test_loader):
            X_test_batch = X_test_batch.to(device)
            y_test_batch = y_test_batch.to(device)
            y_test_batch_bin = y_test_batch_bin.to(device)


            if getattr(args, "ar", 1) > 1:
                B, seq_len, n_features = X_test_batch.shape
                ar_horizon = getattr(args, "ar", 1)

                if i + ar_horizon - 1 >= len(test_loader.dataset):
                    break

                X_ar = X_test_batch.clone()
                preds_ar = torch.zeros(B, ar_horizon).to(device)
                true_ar = torch.zeros(B, ar_horizon).to(device)

                for step in range(ar_horizon):
                    outputs_reg, outputs_cl = model(X_ar)
                    preds_ar[:, step] = outputs_reg.view(-1)

                    # Obtener valor verdadero para ese step
                    true_future = test_loader.dataset[i + step][1]  # index 1 = y_test
                    true_ar[:, step] = torch.tensor(true_future, device=device).view(-1)

                    # Actualizar input autoregresivo
                    new_val = outputs_reg.detach().unsqueeze(1)  # (B,1,1)
                    X_ar = torch.cat([X_ar[:, 1:, :], new_val], dim=1)

                # Guardar resultados
                test_preds_reg.append(preds_ar.cpu())
                test_targets_reg.append(true_ar.cpu())

                # Clasificación corresponde al último horizonte
                last_cl = test_loader.dataset[i + ar_horizon - 1][2]  # index 2 = y_test_bin
                test_preds_cl.append(outputs_cl.cpu())
                test_targets_cl.append(torch.tensor(last_cl.unsqueeze(0)).cpu())
            else:
                # Direct prediction
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
    
    #y_pred_bin = hf.smooth_binary_predictions(y_pred_cl)
    if 'classification' in phases:
        mode, thr_up, thr_down, trend, slope=  hf.grid_search_smoothing(y_true_bin, y_pred_cl)
        y_pred_bin= (y_pred_cl > 0.5).astype(int)
    else: 
        y_pred_bin = hf.ensemble_with_rules(test_preds_reg, scaler_y, threshold, (2/3*threshold))


    
    if 'regression' in phases:
        figure_path_reg = os.path.join(experiment_path, 'test_pred_reg.png')
        hf.plot_y_pred_vs_target(y_true, y_pred_reg, y_pred_bin, y_true, scaler_y, threshold, figure_path=figure_path_reg, task='regression')
    
    figure_path_cl = os.path.join(experiment_path, 'test_pred_cl.png')
    hf.plot_y_pred_vs_target(y_true, y_pred_cl, y_pred_bin, y_true_bin, scaler_y, threshold, figure_path=figure_path_cl, task='classification')
    #figure_path_cl = os.path.join(experiment_path, 'test_pred_cl_ensemble.png')
    #hf.plot_y_pred_vs_target(y_pred_bin, y_true_bin, figure_path=figure_path_cl, task='classification')

    figure_path_pos = os.path.join(experiment_path, 'positive_indices.png')
    hf.plot_positive_indices(y_true_bin, y_pred_bin, figure_path=figure_path_pos)
    #figure_path_pos = os.path.join(experiment_path, 'positive_indices_ensemble.png')
    #hf.plot_positive_indices(y_true_bin, y_pred_bin, figure_path=figure_path_pos)

    # Hard metrics with bootstrap CI
    acc = accuracy_score(y_true_bin, y_pred_bin)
    ci_acc = hf.bootstrap_ci(y_true_bin, y_pred_bin, accuracy_score)
    prec = precision_score(y_true_bin, y_pred_bin)
    ci_prec = hf.bootstrap_ci(y_true_bin, y_pred_bin, lambda y,t: precision_score(y,t,zero_division=0))
    rec = recall_score(y_true_bin, y_pred_bin)
    ci_rec = hf.bootstrap_ci(y_true_bin, y_pred_bin, lambda y,t: recall_score(y,t,zero_division=0))
    f1 = f1_score(y_true_bin, y_pred_bin)
    ci_f1 = hf.bootstrap_ci(y_true_bin, y_pred_bin, lambda y,t: f1_score(y,t,zero_division=0))
    #auc = roc_auc_score(y_true_bin, y_pred_cl)
    #ci_auc = hf.bootstrap_ci(y_true_bin, y_pred_cl, roc_auc_score)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
    fpr = fp / (fp + tn)
    ci_fpr = hf.bootstrap_ci(y_true_bin, y_pred_bin, lambda y,t: confusion_matrix(y,t).ravel()[1]/(confusion_matrix(y,t).ravel()[0]+confusion_matrix(y,t).ravel()[1]))
    brier_score_val = brier_score_loss(y_true_bin, y_pred_cl)
    ci_brier = hf.bootstrap_ci(y_true_bin, y_pred_cl, brier_score_loss)
    
    if 'regression' in phases:
        # Regression metrics with bootstrap CI
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
        #'roc_auc': auc, 'roc_auc_95ci': ci_auc,
        'false_positive_rate': fpr, 'fpr_95ci': ci_fpr,
        'brier_score': brier_score_val, 'brier_95ci': ci_brier,
        'mae': mae, 'mae_95ci': ci_mae,
        'mse': mse, 'mse_95ci': ci_mse,
        'r2_score': r2, 'r2_95ci': ci_r2
    }


    figure_path_soft = os.path.join(experiment_path, 'soft_conf_matrix.png')
    soft_metrics=hf.soft_metrics(y_true_bin, y_pred_cl, figure_path=figure_path_soft, threshold=0.5)

    figure_path = os.path.join(experiment_path, 'fp_fn.png')
    #figure_path_ensemble = os.path.join(experiment_path, 'fp_fn_ensemble.png')
    if 'regression' in phases:
        hf.analyze_classification_errors(y_pred_bin, y_pred_cl, y_true_bin, y_true, y_pred_reg, scaler_y, classifier=False, figure_path=figure_path)
        #hf.analyze_classification_errors(y_pred_bin_ensemble, y_pred_cl, y_true_bin, y_true, y_pred_reg, scaler_y, classifier=False, figure_path=figure_path_ensemble)
    else: 
        hf.analyze_classification_errors(y_pred_bin, y_pred_cl, y_true_bin, y_true, y_pred_reg, scaler_y, classifier=True, figure_path=figure_path)
        #hf.analyze_classification_errors(y_pred_bin_ensemble, y_pred_cl, y_true_bin, y_true, y_pred_reg, scaler_y, classifier=True, figure_path=figure_path_ensemble)

    figure_path_cm = os.path.join(experiment_path, 'confusion_matrix.png')
    hf.conf_matrix(y_true_bin, y_pred_bin, figure_path=figure_path_cm)
    #figure_path_cm = os.path.join(experiment_path, 'confusion_matrix_ensemble.png')
    #hf.conf_matrix(y_true_bin, y_pred_bin_ensemble, figure_path=figure_path_cm)

    figure_path_cl = os.path.join(experiment_path, 'hist_cl.png')
    hf.plot_y_pred_vs_target_hist(y_pred_cl, y_true_bin, y_true, figure_path=figure_path_cl)
    

    #feature_names = ["R", "Rv", "D", "S", "S3", "V3", "S6", "V6", "Sv", "LastRelR", 
    #                 "LastRelS", "CumSum", "R_max", "NumAb075T", "MAratio", "Acc"]
    #feature_names=["Rt", "CumSum", "R_max", "NumAb075T"] #,   "lastrelslope","reldistRt", "relslopeRt"]
    #figure_path_fi = os.path.join(experiment_path, 'feature_ablation_importance.png')
    #hf.feature_ablation_importance(model, test_loader, device, feature_names, figure_path_fi)#, thr_up, thr_down, trend, slope)
    #figure_path_fi = os.path.join(experiment_path, 'feature_addition_importance.png')
    #hf.feature_addition_importance(model, test_loader, device, feature_names, figure_path_fi)#, thr_up, thr_down, trend, slope)
        
    reporter.report('test_metrics', test_metrics)
    reporter.report('soft_metrics', soft_metrics)
    
    figure_path= os.path.join(experiment_path, 'RSvsIA.png')
    ai_rs=hf.comparison_to_rule(args.file_path, model, device, scaler_y, threshold, lookback=args.lookback, forecast=args.forecast, thr_up=thr_up, thr_down=thr_down, trend=trend, slope_rel=slope, mode=mode, figure_path=figure_path)
    #ai_rs=hf.comparison_to_rule(args.file_path, model, device, scaler_y, threshold, lookback=args.lookback, forecast=args.forecast, figure_path=figure_path)
    reporter.report('cost_safety',ai_rs)
    
        
    print("\nTraining and evaluation complete.")


if __name__ == "__main__":
    main()