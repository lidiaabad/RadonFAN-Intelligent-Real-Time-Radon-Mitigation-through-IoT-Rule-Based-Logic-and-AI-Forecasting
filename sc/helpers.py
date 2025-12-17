import matplotlib.pyplot as plt
import numpy as np
import random
import json
import time
import re
from sklearn.manifold import TSNE
#from torchinfo import summary
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import torch
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
from thop import profile, clever_format  
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from datapreprocesing import preprocess_data, create_dataloaders

'''
This is a script used to accomodate helper functions for the training and testing pipeline
'''

'''
========================================= BASIC FUNCTIONS ========================================
'''
    
def set_seed(seed):
    '''
    Helper function to set seed
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    random.seed(seed)

def get_forward_time(model, device):
    '''
    Helper function to get forward time
    '''
    model.to('cpu')
    dummy_data = torch.rand(1, 24000)
    start = time.time()
    _ = model(dummy_data)
    end = time.time()
    forward_time = end - start
    model.to(device)
    return forward_time

'''
def get_model_summary(model, device, input_size=(1, 512, 1)):
    Helper function to show model summary

    model_input = torch.randn(input_size).to(device)
    info = summary(model, input_data=[model_input], verbose=1)
    model_summary = {
        "model": str(model),
        "total_parameters": info.total_params,
        "trainable_parameters": info.trainable_params}

    return model_summary
'''

def get_computational_cost(model, input_shapes): 
    '''
    Helper to compute and print model computational cost (params, FLOPs, size).
    
    Args:
        model: nn.Module
        input_shapes: tuple or list of input tensors for FLOPs computation
    Returns:
        dict with 'params', 'trainable_params', 'model_size_MB', 'FLOPs'
    '''

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 ** 2)  # float32 → 4 bytes

    # Compute FLOPs (approx)
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=input_shapes, verbose=False)
        flops, params = clever_format([macs * 2, params], "%.3f")  # FLOPs ≈ 2 × MACs

    # Print summary
    print("─────────────────────────────────────────────")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size:           {model_size_mb:.2f} MB")
    print(f"FLOPs (per forward):  {flops}")
    print("─────────────────────────────────────────────")

    # Return results for storage
    stats = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_MB": model_size_mb,
        "FLOPs": flops
    }

    return stats

'''
========================================== PLOT FUNCTIONS =========================================
'''

def save_train_metrics_plot(metrics, figure_path, phase=None):
    '''
    Helper function to plot the training and validation losses per epoch
    Input: 
        - metrics dicitonary with all loses per epoch
        - figure path
        - phase if we want to plot only regresion; only classification or combined losses
    Output: 
        - figure
    '''
    plt.figure(figsize=(8, 4))

    if phase == 'regression':
        plt.plot(metrics['train_loss_reg'], label='Training regression loss')
        plt.plot(metrics['validation_loss_reg'], label='Validation regression loss')

    elif phase == 'classification':
        plt.plot(metrics['train_loss_cl'], label='Training classification loss')
        plt.plot(metrics['validation_loss_cl'], label='Validation classification loss')

    else:  # fallback: combined view
        plt.plot(metrics['train_loss'], label='Training loss')
        plt.plot(metrics['validation_loss'], label='Validation loss')
        plt.plot(metrics['train_loss_reg'], label='Training regression loss')
        plt.plot(metrics['validation_loss_reg'], label='Validation regression loss')
        plt.plot(metrics['train_loss_cl'], label='Training classification loss')
        plt.plot(metrics['validation_loss_cl'], label='Validation classification loss')

    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    
def plot_discriminator_loss(losses, losses_val, save_path=None):
    '''
    Helper function to plot discirminator train and val losses
    Input: 
        - list with train and validation losses
        - figure path
    Output: 
        - figure
    '''
    plt.figure(figsize=(6, 4))
    plt.plot(losses, label='Training Loss')
    plt.plot(losses_val, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_weighted_conf_matrix(weighted_tp, weighted_fp, weighted_fn, weighted_tn, figure_path=None):
    '''
    Helper function to plot the confidence matric for the soft metrics
    Input: 
        - Soft TP, FP, FN and TN
        - figure path to store the figure
    Output: 
        - figure
    '''
    cm = np.array([[weighted_tn, weighted_fp],
                   [weighted_fn, weighted_tp]])
    
    cm_agg = cm.sum(axis=2) if cm.ndim == 3 else cm
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_agg, annot=True, fmt=".2f", cmap="Blues", xticklabels=['Pred 0','Pred 1'], yticklabels=['True 0','True 1'])
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(figure_path, bbox_inches='tight')

def conf_matrix(y_true, y_pred_bn, figure_path):
    '''
    Helper function to plot the confusion matrix
    Input: 
        - target and predicted binary values
        - figure path
    Output: 
        - figure
    '''
    cm = metrics.confusion_matrix(y_true, y_pred_bn)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label", rotation=90)
    plt.savefig(figure_path)
    plt.close()

def plot_y_pred_vs_target(y_reg, y_pred_tensor, y_pred_bin, y_target_tensor, scaler, threshold, slope=0.3, figure_path=None, task="regression"):
    '''
    Helper function to plot for the test set the target and predicted values in regression/classification task
    Input: 
        - predicted and real values
        - figure path
        - task: regression or classification
    Output: 
        - figure
    '''

    y_pred = y_pred_tensor.detach().cpu().numpy() if isinstance(y_pred_tensor, torch.Tensor) else np.array(y_pred_tensor)
    y_target = y_target_tensor.detach().cpu().numpy() if isinstance(y_target_tensor, torch.Tensor) else np.array(y_target_tensor)
    
    indices = np.arange(len(y_pred))
        
    if task == 'regression':

        H = y_pred.shape[1]
        # 3 plots: standardized values against index;  residuals against index, residuals against true value
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(21, 10), sharex=False)
        cmap = plt.get_cmap("Set3")
        colors = cmap.colors
        
        # --- Standardized values against index
        # we plot for each horizon: firs the biggest horizon as it is the most different one
        for h in reversed(range(H)):        
            shifted_indices = indices + h
            ax1.plot(shifted_indices, y_pred[:, h], color=colors[h], label=f'Pred t+{h+1}')
            
        ax1.plot(indices, y_target[:, 0], label='True', color='black', linewidth=1.0)
        ax1.legend()
        ax2.set_xlabel('Sample Index')
        ax1.set_ylabel('Normalized Radon Value')

        # --- Residuals against index 
        # we plot for each horizon: firs the biggest horizon as it is the most different one
        for h in reversed(range(H)):
            error = y_pred[:, h] - y_target[:, h]
            ax2.plot(indices, error, color=colors[h], label=f'Err t+{h+1}')
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.legend()
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Residual (y_pred − y_target)')


    elif task == 'classification':
        
        
        N = y_reg.shape[0]
        steps = y_reg.shape[1]
        
        if y_reg.ndim == 3:
            vals = y_reg.shape[2]
            y_reshape = y_reg.reshape(-1, vals)[:, 0]  # flatten for scaler
        else:
            vals = 1
            y_reshape = y_reg.reshape(-1, 1)

        y_inv = scaler.inverse_transform(y_reshape)
        y_inv=y_inv.reshape(N, steps)
        
        y_bin_RS = np.zeros_like(y_target)
        
        # Cálculo de diferencias correctamente (derivada aproximada)
        diff = np.zeros_like(y_inv[:, 0])  # suponiendo y_reg.shape = (N, 6)
        diff[1:] = (y_inv[1:, 0] - y_inv[:-1, 0]) / 10.0  # derivada sobre el primer canal

        # Generación de la señal binaria del sistema de reglas
        for i in range(1, len(y_inv)): 
            if (y_inv[i, 0] > threshold or 
                (y_inv[i, 0] > (2/3) * threshold and diff[i] > slope)): 
                y_bin_RS[i] = 1
            elif y_inv[i, 0] > (2/3) * threshold and diff[i] > -slope: 
                y_bin_RS[i] = y_bin_RS[i-1]
                
        
        y_pred=y_pred.flatten()
        y_target=y_target.flatten()
        
        error = y_pred - y_target
        
        # 2 plots: values and residuals agains index
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18,12), sharex=False)
        
        # --- Values against index
        ax1.plot(indices, y_bin_RS, label='RS Value', color='tab:purple', linewidth=1.0, alpha=0.5)
        ax1.plot(indices, y_pred_bin, label='Predicted Binary Value', color='tab:orange', linewidth=1.0, alpha=0.5)
        ax1.plot(indices, y_target, label='True Label', color='tab:green', linewidth=1.0)
        ax1.plot(indices, y_pred, label='Predicted Value', color='tab:blue', linewidth=1.0)

        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Classification value')
        ax1.legend()
        
        ax2.plot(indices, y_pred_bin, label='Predicted Binary Value', color='tab:orange', linewidth=1.0, alpha=0.5)
        ax2.plot(indices, y_target, label='True Label', color='tab:green', linewidth=1.0)
        ax2.plot(indices, y_pred, label='Predicted Value', color='tab:blue', linewidth=1.0)

        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Classification value')
        ax2.legend()
        ax2.set_ylim(0.3, 0.7)
        

        # Residuals agains index
        ax3.plot(indices, error, label='Residual value', color='tab:red', alpha=0.6)
        ax3.axhline(0, linestyle='--', color='gray')
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('Residual (y_pred − y_target)')
        ax3.legend()

    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    
def classes_sep(X_before, X_after, y_np, epoch, figure_path=None): 
    '''
    Helper function to see the separation between classes at the beginning and end of the classification head
    Input: 
        - hookers before and after the classification head
        - target labels
        - number of epoch
        - figure path
    Output: 
        - n figures for n epochs
    '''
    
    Xb_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_before)
    Xa_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_after)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.scatter(Xb_2d[:,0], Xb_2d[:,1], c=y_np, cmap="coolwarm", s=5)
    plt.title(f"Antes del MLP (Epoch {epoch})")

    plt.subplot(1,2,2)
    plt.scatter(Xa_2d[:,0], Xa_2d[:,1], c=y_np, cmap="coolwarm", s=5)
    plt.title(f"Después del MLP (Epoch {epoch})")

    plt.tight_layout()

    if figure_path is not None:
        plt.savefig(figure_path, dpi=300)
        plt.close()
    else:
        plt.show()
     
def plot_attn(attn_weights, epoch=None, figure_path=None, average_heads=True):
    '''
    Helper function to analyze the attention weights of the future time steps (query) 
    w/ respect to past sequence (key)
    Input: 
        - attention weight
        - epoch
        - figure path
        - average heads to perform mean if several heads
    Output: 
        - n figures for n epochs
    '''
    attn = attn_weights.detach().cpu().numpy()  
    if average_heads:
        attn_to_plot = attn.mean(axis=0) 
    else:
        attn_to_plot = attn     
    plt.clf()
    plt.imshow(attn_to_plot, aspect="auto", cmap="viridis")
    plt.colorbar(label="Attention weight")
    plt.xlabel("Input timestep (T)")
    plt.ylabel("Forecast step (F)")
    plt.title(f"Attention weights - epoch {epoch}")
    plt.savefig(figure_path)
    plt.close()
              
def plot_metrics_two_subplots(fp_packs, fn_packs, figure_path):
    '''
    Helper function to plot the FP and FN true values for each horizon
    Input: 
        - FP and FN pachs
        - figure path
    Output: 
        - figure
    '''
    cmap = plt.get_cmap("Set3")
    colors = cmap.colors
    horizons = [f't+{i+1}' for i in range(fp_packs.shape[1])]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # weighted_fp subplot
    for i in reversed(range(fp_packs.shape[1])):
        sns.kdeplot(fp_packs[:, i], label=horizons[i], color=colors[i], fill=True, alpha=0.5, bw_adjust=0.5, ax=axes[0])
    axes[0].set_xlabel("False Positive Value")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # weighted_fn subplot
    for i in reversed(range(fn_packs.shape[1])):
        sns.kdeplot(fn_packs[:, i], label=horizons[i], color=colors[i], fill=True, alpha=0.5, bw_adjust=0.5, ax=axes[1])
    axes[1].set_xlabel("False Negative Value")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()

def plot_horizon_kdes(fp_horizons, fn_horizons, forecast_len, figure_path):
    '''
    Helper function to plot a kde indicating for which horizon we get FP or FN
    Input: 
        - horizons for FP and FN
        - forecast len
        - figure path
    Output: 
        - figure
    '''
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # False positives
    sns.kdeplot(fp_horizons, fill=True, alpha=0.4, bw_adjust=0.3,
                ax=axes[0], color="tab:blue")
    axes[0].set_xticks(range(forecast_len))
    axes[0].set_xticklabels([f"t+{i+1}" for i in range(forecast_len)])
    axes[0].set_xlabel("Horizon False Positive")
    axes[0].set_ylabel("Density")

    # False negatives
    sns.kdeplot(fn_horizons, fill=True, alpha=0.4, bw_adjust=0.3,
                ax=axes[1], color="tab:red")
    axes[1].set_xticks(range(forecast_len))
    axes[1].set_xticklabels([f"t+{i+1}" for i in range(forecast_len)])
    axes[1].set_xlabel("Horizon False Negative")

    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()

def plot_fp_fn_scatter_from_packs(fp_packs, fn_packs, y_pred_inv, fp_mask, fn_mask, figure_path):
    '''
    Helper fucntion to plot the scatter predicted values for FP and FN with colors according to horizon
    Input: 
        - FP and FN pachs
        - predicted values (real scale)
        - FP and FN mask 
        - figure path
    Output: 
        - figure
    '''
    
    fp_pred_packs = y_pred_inv[fp_mask, :]
    fn_pred_packs = y_pred_inv[fn_mask, :]

    # Si ambos están vacíos, no hacemos nada
    if fp_packs.size == 0 and fn_packs.size == 0:
        print("No hay falsos positivos ni falsos negativos para graficar.")
        return

    # n_time_steps solo tiene sentido si hay algo
    n_time_steps = fp_packs.shape[1] if fp_packs.size > 0 else fn_packs.shape[1]

    cmap = plt.get_cmap("viridis", n_time_steps)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    # False Positives
    if fp_packs.size > 0 and fp_pred_packs.size > 0:
        for t in range(n_time_steps):
            sns.scatterplot(
                x=fp_packs[:, t], y=fp_pred_packs[:, t],
                ax=axes[0], alpha=0.6, s=20, edgecolor=None,
                color=cmap(t), label=f"t+{t+1}"
            )
        min_val = min(fp_packs.min(), fp_pred_packs.min())
        max_val = max(fp_packs.max(), fp_pred_packs.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
        axes[0].set_title("False Positives")
        axes[0].set_xlabel("True Value")
        axes[0].set_ylabel("Predicted Value")
        axes[0].legend(title="Time step")
    else:
        axes[0].set_title("False Positives (none)")
        axes[0].axis("off")

    # False Negatives
    if fn_packs.size > 0 and fn_pred_packs.size > 0:
        for t in range(n_time_steps):
            sns.scatterplot(
                x=fn_packs[:, t], y=fn_pred_packs[:, t],
                ax=axes[1], alpha=0.6, s=20, edgecolor=None,
                color=cmap(t), label=f"t+{t+1}"
            )
        min_val = min(fn_packs.min(), fn_pred_packs.min())
        max_val = max(fn_packs.max(), fn_pred_packs.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
        axes[1].set_title("False Negatives")
        axes[1].set_xlabel("True Value")
        axes[1].legend(title="Time step")
    else:
        axes[1].set_title("False Negatives (none)")
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
       
def plot_positive_indices(y_true_bin, y_pred_bin, figure_path):
    '''
    Helper function to plot the TP (green), FP (red) and FN (blue) indices. 
    Input: 
        - true and predicted binary values
        - figure path
    Output: 
        - figure
    '''
    # Indices where at least one is positive
    indices = np.where((y_true_bin == 1) | (y_pred_bin == 1))[0]

    # Categories
    both = indices[(y_true_bin[indices] == 1) & (y_pred_bin[indices] == 1)]
    true_only = indices[(y_true_bin[indices] == 1) & (y_pred_bin[indices] == 0)]
    pred_only = indices[(y_true_bin[indices] == 0) & (y_pred_bin[indices] == 1)]

    # Split into 4 roughly equal chunks
    chunks = np.array_split(indices, 4)

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharey=True)

    for i, ax in enumerate(axes):
        sub_idx = chunks[i]

        both_sub = [idx for idx in sub_idx if idx in both]
        true_only_sub = [idx for idx in sub_idx if idx in true_only]
        pred_only_sub = [idx for idx in sub_idx if idx in pred_only]

        # Add small vertical jitter to avoid overlapping points
        jitter = 0.1
        ax.scatter(true_only_sub, 1 + np.random.uniform(-jitter, jitter, len(true_only_sub)), 
                   color="blue", s=20, alpha=0.7, label="True=1 only")
        ax.scatter(pred_only_sub, 2 + np.random.uniform(-jitter, jitter, len(pred_only_sub)), 
                   color="red", s=20, alpha=0.7, label="Pred=1 only")
        ax.scatter(both_sub, 3 + np.random.uniform(-jitter, jitter, len(both_sub)), 
                   color="green", s=20, alpha=0.7, label="Both=1")

        # Reduce x-tick density
        if len(sub_idx) > 20:
            xticks = sub_idx[::max(1, len(sub_idx)//20)]
        else:
            xticks = sub_idx
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, rotation=90, fontsize=8)
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["True=1 only", "Pred=1 only", "Both=1"])
        ax.set_xlabel("Index")
        if i == 0:
            ax.set_ylabel("Category")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(figure_path)
    plt.close()
   
def plot_regression_weights(y_reg, y_bin, scaler,  figure_path, threshold=100, margin=0.9, 
                                 low_frac=0.25, sigma_low_frac=0.4, sigma_high_frac=0.2, forecast=6,
                                 boost_weight=10.0, window=6):
    
    '''
    Helper function to plot the distribution of regression weights according to distance from threshold
    Input: 
        - y_train values
        - figure path
        - scaler to perform inverse transform
        - threshold 
        - lower, upper: llimits of upper region
        - steepness: how fast ouutside
    Output: 
        - figure
        - weights
    '''

    # --- Desnormalizamos todos los valores
    y_orig = scaler.inverse_transform(y_reg)

    # --- Tomamos SOLO el último valor de cada fila
    last_vals = y_orig[:, -1]  # shape (N,)

    # --- Parámetros de control
    low_limit = low_frac * threshold
    sigma = low_frac * threshold

    # --- Gaussiana centrada en threshold
    weights = np.exp(-0.5 * ((last_vals - (threshold*0.9)) / sigma) ** 2)

    # --- Penalización fuerte para valores muy bajos
    low_mask = last_vals < low_limit
    weights[low_mask] *= 0.05

    # --- Refuerzo para valores dentro del rango [0.8, 1.2] * threshold
    near_mask = (last_vals >= 0.5* threshold) & (last_vals <= 1.5 * threshold)
    boost = 1 + (boost_weight - 1) * np.exp(-0.5 * ((last_vals - (threshold*0.9)) / (0.2 * threshold)) ** 2)
    weights[near_mask] *= boost[near_mask]
    
    seq_multiplier = np.ones_like(weights, dtype=float)

    # Convertimos y_bin a 1D si viene por columnas (e.g. (N,1))
    y_bin_flat = y_bin.reshape(-1).astype(int)

    count = 0
    for i in range(len(y_bin_flat)):
        if y_bin_flat[i] == 1:
            if i == 0 or y_bin_flat[i - 1] == 0:
                # Primer 1 después de 0 → forecast
                count = forecast
            else:
                # Segundo, tercero... va bajando
                count = max(count - 1, 1)
            seq_multiplier[i] = count
        else:
            # Reinicia la secuencia si hay un 0
            count = 0

    # --- Aplicamos el multiplicador ---
    weights *= seq_multiplier


    # === Plot ===
    plt.figure(figsize=(6, 5))
    plt.scatter(last_vals, weights, alpha=0.5, s=10, color='teal')
    #plt.axvline(threshold, color='black', lisnestyle='-.', label='T')
    plt.xlabel('y$_{t+6}$ (Bq/m$^3$)')
    plt.ylabel('Sample Weight')
    #plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()

def plot_y_pred_vs_target_hist(y_pred_cl, y_true_bin, y_true, figure_path=None):
    """
    Plots KDEs of prediction scores separated by binary target labels.
    
    Parameters
    ----------
    y_pred_cl : array-like
        Model predicted scores or logits.
    y_true_bin : array-like
        Binary ground truth labels (0 or 1).
    figure_path : str, optional
        Path to save the figure. If None, the plot is just displayed.
    """
    
    y_pred_cl = np.asarray(y_pred_cl)
    y_true_bin = np.asarray(y_true_bin)
    y_true = np.asarray(y_true)

    # Get last value of y_true (assuming shape [samples, time])
    if y_true.ndim > 1:
        y_last = y_true[:, -1]
    else:
        y_last = y_true

    preds_pos = y_pred_cl[y_true_bin == 1]
    preds_neg = y_pred_cl[y_true_bin == 0]
    y_last_pos = y_last[y_true_bin == 1]
    y_last_neg = y_last[y_true_bin == 0]

    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=False)

    # --- (1) KDE plot ---
    sns.kdeplot(preds_neg, fill=True, color='blue', alpha=0.4, label='y_true = 0', ax=axes[0])
    sns.kdeplot(preds_pos, fill=True, color='red', alpha=0.4, label='y_true = 1', ax=axes[0])
    axes[0].axvline(0.4, color='gray', linestyle='--')
    axes[0].axvline(0.6, color='gray', linestyle='--')
    axes[0].axvline(0.5, color='black', linestyle='--')
    axes[0].set_xlabel('Predicted value')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    # --- (2) Scatter vs last y_true ---
    axes[1].scatter(preds_neg, y_last_neg, color='blue', alpha=0.4, label='y_true_bin = 0')
    axes[1].scatter(preds_pos, y_last_pos, color='red', alpha=0.4, label='y_true_bin = 1')
    axes[1].axvline(0.4, color='gray', linestyle='--')
    axes[1].axvline(0.6, color='gray', linestyle='--')
    axes[1].axvline(0.5, color='black', linestyle='--')
    axes[1].set_xlabel('Last target value')
    axes[1].set_ylabel('Predicted value')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()

def feature_ablation_importance(model, test_loader, device, feature_names, figure_path):
    model.eval()
    X_list, y_list = [], []
    with torch.no_grad():
        for Xb, _, yb_bin, _, _ in test_loader:
            X_list.append(Xb.cpu().numpy())
            y_list.append(yb_bin.cpu().numpy())
    X = np.concatenate(X_list)
    y = np.concatenate(y_list).ravel()
    medians = np.nanmedian(X.reshape(-1, X.shape[-1]), axis=0)

    def predict_from_array(arr):
        preds = []
        with torch.no_grad():
            for i in range(0, len(arr), test_loader.batch_size):
                xb = torch.tensor(arr[i:i+test_loader.batch_size]).float().to(device)
                out = model(xb)
                logits = out[1] if isinstance(out, tuple) else out
                preds.append(logits.cpu().numpy().ravel())
        return np.concatenate(preds)

    baseline_logits = predict_from_array(X)
    #ybin_base = smooth_binary_predictions(baseline_logits, thrup, thrlow, trend, slope, mode="hysteresis")
    ybin_base = (baseline_logits > 0.5).astype(int)
    tn, fp_base, fn_base, tp = confusion_matrix(y, ybin_base).ravel()
    print(f"[BASELINE] FP={fp_base}, FN={fn_base}")

    records = []
    for fi, fname in enumerate(feature_names):
        X_mod = X.copy()
        X_mod[..., fi] = medians[fi]
        preds_mod = predict_from_array(X_mod)

        #ybin_mod = smooth_binary_predictions(preds_mod, thrup, thrlow, trend, slope, mode="hysteresis")
        ybin_mod = (preds_mod > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, ybin_mod).ravel()
        delta_logit = np.mean(np.abs(preds_mod - baseline_logits))

        records.append({
            "feature": fname,
            "delta_fp": fp - fp_base,
            "delta_fn": fn - fn_base,
            "delta_logit": delta_logit
        })

    df = pd.DataFrame(records).sort_values("delta_fn", ascending=False)

    # ---- Plots ----
    metrics = ["delta_fp", "delta_fn", "delta_logit"]
    titles = ["ΔFP (feature ablation)", "ΔFN (feature ablation)", "ΔLogit (feature ablation)"]

    fig, axes = plt.subplots(1, 3, figsize=(18, len(df)*0.4), sharey=True)
    for ax, metric, title in zip(axes, metrics, titles):
        colors = ["#3b82f6" if v>0 else "#ef4444" for v in df[metric]]
        ax.barh(df["feature"], df[metric], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()
    return df

def feature_addition_importance(model, test_loader, device, feature_names, figure_path, neutral="median"):

    model.eval()
    X_list, y_list = [], []
    with torch.no_grad():
        for Xb, _, yb_bin, _, _ in test_loader:
            X_list.append(Xb.cpu().numpy())
            y_list.append(yb_bin.cpu().numpy())
    X = np.concatenate(X_list)
    y = np.concatenate(y_list).ravel()
    medians = np.nanmedian(X.reshape(-1, X.shape[-1]), axis=0)
    neutral_vals = medians if neutral=="median" else np.min(X.reshape(-1, X.shape[-1]), axis=0)

    def predict_from_array(arr):
        preds = []
        with torch.no_grad():
            for i in range(0, len(arr), test_loader.batch_size):
                xb = torch.tensor(arr[i:i+test_loader.batch_size]).float().to(device)
                out = model(xb)
                logits = out[1] if isinstance(out, tuple) else out
                preds.append(logits.cpu().numpy().ravel())
        return np.concatenate(preds)

    X_base = np.tile(neutral_vals, (X.shape[0], X.shape[1],1))
    baseline_logits = predict_from_array(X_base)
    #ybin_base = smooth_binary_predictions(baseline_logits, thrup, thrlow, trend, slope, mode="hysteresis")
    ybin_base = (baseline_logits > 0.5).astype(int)
    tn, fp_base, fn_base, tp = confusion_matrix(y, ybin_base).ravel()
    print(f"[NEUTRAL BASELINE] FP={fp_base}, FN={fn_base}")

    records=[]
    for fi, fname in enumerate(feature_names):
        X_mod = X_base.copy()
        X_mod[..., fi] = X[..., fi]  # restore feature fi
        preds_mod = predict_from_array(X_mod)

        #ybin_mod = smooth_binary_predictions(preds_mod, thrup, thrlow, trend, slope, mode="hysteresis")
        ybin_mod = (preds_mod > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, ybin_mod).ravel()
        delta_logit = np.mean(np.abs(preds_mod - baseline_logits))

        records.append({
            "feature": fname,
            "delta_fp": fp - fp_base,
            "delta_fn": fn - fn_base,
            "delta_logit": delta_logit
        })

    df = pd.DataFrame(records).sort_values("delta_fn", ascending=True)

    # ---- Plots ----
    metrics = ["delta_fp", "delta_fn", "delta_logit"]
    titles = ["ΔFP (feature addition)", "ΔFN (feature addition)", "ΔLogit (feature addition)"]

    fig, axes = plt.subplots(1, 3, figsize=(18, len(df)*0.4), sharey=True)
    for ax, metric, title in zip(axes, metrics, titles):
        colors = ["#3b82f6" if v<0 else "#ef4444" for v in df[metric]]  # negativo = mejora
        ax.barh(df["feature"], df[metric], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()
    return df

'''
========================================  OTHER FUNCTIONS  =======================================
'''

def get_metrics(packs_true, cl_pred, pack_type="FP", packs_pred= None):
    '''
    Helper function to print information about the real and predicted values of FP and FN instances
    '''
    if packs_pred is not None: 
        for i, (pack_true, pack_pred) in enumerate(zip(packs_true, packs_pred)):
            print(f"{pack_type} pack {i}: true= {pack_true}, pred = {pack_pred} with logit {cl_pred[i]}")
    else: 
        for i, pack_true in enumerate(packs_true):
            print(f"{pack_type} pack {i}: true= {pack_true} with logit {cl_pred[i]}")

def get_error_horizons(y_true_reg, y_pred_reg, mask, thresholds, mode="fp"):
    '''
    Helper function to get the horizon at which we have a FP and FN; i.e. if responsible
    step is [+1... +6]
    Input: 
        - true and predicted ergression values
        - FP or FN mask
        - threhsolds
        - mode
    Output: 
        - horizons 
    '''

    thresholds = np.array(thresholds)
    horizons = []
    for yt, yp in zip(y_true_reg[mask], y_pred_reg[mask]):
        cond_true = yt > thresholds
        cond_pred = yp > thresholds
        if mode == "fp":
            idxs = np.where(cond_pred & ~cond_true)[0]
        elif mode == "fn":
            idxs = np.where(~cond_pred & cond_true)[0]
        else:
            raise ValueError("mode must be 'fp' or 'fn'")
        horizons.extend(idxs)
    return np.array(horizons)

def analyze_classification_errors(y_pred_bin, y_pred_proba, y_true_cls, y_true_reg, y_pred_reg, scaler, classifier, figure_path):
    '''
    Helper function to show information related to the classiication errors: get values of FP and FN; print and
    plot information of true and predicted FP and FN values; analysis of problematic horizons and plot of values 
    per horizon
    Input: 
        - prediced and true regression and classification values
        - scaler for inverse transform
        - classifier flag to not attend to regression analysis
        - figure path
    Output:
        - several prints and plots with FP and FN information 
    '''

    # Boolean masks
    fp_mask = (y_pred_bin == 1) & (y_true_cls == 0)
    fn_mask = (y_pred_bin == 0) & (y_true_cls == 1)
    '''
    def expand_mask(mask, window):
        expanded = mask.copy()
        for i in range(1, window+1):
            expanded[i:] |= mask[:-i]  # desplaza hacia abajo y hace OR
        return expanded

    fp_mask= expand_mask(fp_mask, 3)
    fn_mask = expand_mask(fn_mask, 3)
    '''
    
    if y_true_reg.ndim == 1:
        y_true_reg = y_true_reg.reshape(-1, 1)
    N, steps = y_true_reg.shape
    y_reshape = y_true_reg.reshape(-1, 1)  # N*6, 1
    y_inv = scaler.inverse_transform(y_reshape)
    y_inv=y_inv.reshape(N, steps)
    
    fp_packs = y_inv[fp_mask, :]  # shape (n_fp, 6)
    fn_packs = y_inv[fn_mask, :]  # shape (n_fn, 6)
    fp_cl_pred = y_pred_proba[fp_mask]  # shape (n_fp)
    fn_cl_pred = y_pred_proba[fn_mask]  # shape (n_fn)
    
    if not classifier: 
        if y_pred_reg.ndim == 1:
            y_pred_reg = y_pred_reg.reshape(-1, 1)
        y_pred_reg_reshape = y_pred_reg.reshape(-1, 1)  # N*6, 1
        y_inv_pred = scaler.inverse_transform(y_pred_reg_reshape)
        y_inv_pred=y_inv_pred.reshape(N, steps)

        fp_packs_pred = y_inv_pred[fp_mask, :]  # shape (n_fp, 6)
        fn_packs_pred = y_inv_pred[fn_mask, :]  # shape (n_fn, 6)
        
        get_metrics(fp_packs, fp_cl_pred, "FP", fp_packs_pred)
        get_metrics(fn_packs, fn_cl_pred, "FN", fn_packs_pred)
    
    else: 
        get_metrics(fp_packs, fp_cl_pred, "FP")
        get_metrics(fn_packs, fn_cl_pred, "FN")
        
    
    plot_metrics_two_subplots(fp_packs, fn_packs, figure_path)

    # BEFORE: a quitar
    forecast_len = steps
    thresholds = [250 if i < 3 else 300 for i in range(forecast_len)]

    if not classifier: 
        y_pred_reshape = y_pred_reg.reshape(-1, 1)  # N*6, 1
        y_inv_pred = scaler.inverse_transform(y_pred_reshape)
        y_inv_pred=y_inv_pred.reshape(N, steps) #respahe to N, 6
        # Obtener horizontes con error
        figure_path_horizons= figure_path.replace(".png", "_horizons.png")
        fp_horizons = get_error_horizons(y_inv, y_inv_pred, fp_mask, thresholds, mode="fp")
        fn_horizons = get_error_horizons(y_inv, y_inv_pred, fn_mask, thresholds, mode="fn")

        plot_horizon_kdes(fp_horizons, fn_horizons, forecast_len, figure_path_horizons)
        
        figure_path_scatter= figure_path.replace(".png", "_scatter.png")
        plot_fp_fn_scatter_from_packs(fp_packs, fn_packs, y_inv_pred, fp_mask, fn_mask, figure_path_scatter)
        
def soft_metrics(y_true_bin, y_pred_prob, figure_path=None,
                 threshold=0.5, weights=[1.0, 0.75, 0.5, 0.25],
                 n_boot=1000, ci=0.95):
    '''
    Helper function to compute soft metrics. Inspired by: 
    SoftED: Metrics for soft evaluation of time series event detection (https://doi.org/10.1016/j.cie.2024.110728)
    Input: 
        - binary target
        - predicted probability (after sigmoid)
        - figure path to save confusion matrix
        - threshold to define y=1 or y=0
        - weights for TP membership according to shift
        - number of bootstrap and confidence of intervals: to allow for bottstrap call
    Output: 
        - soft supervised classification metrics 
        - CI of each metric
        - confusion matrix
    '''
    y_true_bin = np.array(y_true_bin)
    y_pred_prob = np.array(y_pred_prob)
    n = len(y_true_bin)
    max_shift = len(weights) - 1

    # Hard binary predictions
    y_pred_bin = (y_pred_prob >= threshold).astype(int)

    # --- Compute soft TP/FP/FN/TN with ordered matching ---
    matched_pred = np.zeros_like(y_pred_bin, dtype=bool)
    tp_soft = 0.0

    # --- First find match in exact positions; then withshift 1, 2... max
    #     if match: assign weight given by the weight list 
    for shift in range(0, max_shift + 1):  # start with 0, then expand outward
        for i, y in enumerate(y_true_bin):
            if y == 1:
                for sign in [-1, 1] if shift > 0 else [0]:
                    idx = i + sign * shift
                    if 0 <= idx < n and y_pred_bin[idx] == 1 and not matched_pred[idx]:
                        tp_soft += weights[shift]
                        matched_pred[idx] = True
                        break  
    
    # --- Once we find corresponding match to true possition compute the
    #     remaining classes using total number of true and predicted vals
    tp_soft = min(tp_soft, y_true_bin.sum())                 
    fn_soft = y_true_bin.sum() - tp_soft
    fp_soft = y_pred_bin.sum() - tp_soft
    tn_soft = (n - y_true_bin.sum()) - fp_soft

    # Soft metrics
    precision = tp_soft / (tp_soft + fp_soft + 1e-8)
    recall = tp_soft / (tp_soft + fn_soft + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp_soft + tn_soft) / n
    fpr = fp_soft / (fp_soft + tn_soft + 1e-8)
    auc_soft = roc_auc_score(y_true_bin, y_pred_prob) if len(np.unique(y_true_bin)) > 1 else np.nan

    results = {
        "accuracy_soft": float(accuracy),
        "precision_soft": float(precision),
        "recall_soft": float(recall),
        "f1_soft": float(f1),
        "fpr_soft": float(fpr),
        "auc_soft": float(auc_soft) if not np.isnan(auc_soft) else None,
        "tp_soft": float(tp_soft),
        "fp_soft": float(fp_soft),
        "fn_soft": float(fn_soft),
        "tn_soft": float(tn_soft),
    }

    # --- Bootstrap confidence intervals ---
    if n_boot:
        metric_names = ["accuracy_soft", "precision_soft", "recall_soft", "f1_soft", "auc_soft"]
        ci_low = (1 - ci) / 2
        ci_high = 1 - ci_low

        for metric_name in metric_names:
            values = []
            for _ in range(n_boot):
                idxs = np.random.choice(n, n, replace=True)
                yt_sample = y_true_bin[idxs]
                yp_sample = y_pred_prob[idxs]

                # Recompute soft metrics with matched_pred
                matched_sample = np.zeros_like(yt_sample, dtype=bool)
                tp_s = 0.0
                y_pred_bin_sample = (yp_sample >= threshold).astype(int)
                for i, y in enumerate(yt_sample):
                    if y == 1:
                        for shift in range(-max_shift, max_shift + 1):
                            idx = i + shift
                            if 0 <= idx < len(yt_sample) and y_pred_bin_sample[idx] == 1 and not matched_sample[idx]:
                                tp_s += weights[abs(shift)]
                                matched_sample[idx] = True
                                break
                fn_s = yt_sample.sum() - tp_s
                fp_s = y_pred_bin_sample.sum() - matched_sample.sum()
                tn_s = len(yt_sample) - yt_sample.sum() - fp_s

                if metric_name == "accuracy_soft":
                    val = (tp_s + tn_s) / len(yt_sample)
                elif metric_name == "precision_soft":
                    val = tp_s / (tp_s + fp_s + 1e-8)
                elif metric_name == "recall_soft":
                    val = tp_s / (tp_s + fn_s + 1e-8)
                elif metric_name == "f1_soft":
                    prec_s = tp_s / (tp_s + fp_s + 1e-8)
                    rec_s = tp_s / (tp_s + fn_s + 1e-8)
                    val = 2 * prec_s * rec_s / (prec_s + rec_s + 1e-8)
                elif metric_name == "auc_soft":
                    val = roc_auc_score(yt_sample, yp_sample) if len(np.unique(yt_sample)) > 1 else np.nan
                else:
                    val = np.nan
                values.append(val)

            values_clean = np.array([v for v in values if not np.isnan(v)])
            if len(values_clean) == 0:
                results[f"{metric_name}_ci_lower"] = results[metric_name]
                results[f"{metric_name}_ci_upper"] = results[metric_name]
            else:
                results[f"{metric_name}_ci_lower"] = float(np.percentile(values_clean, ci_low * 100))
                results[f"{metric_name}_ci_upper"] = float(np.percentile(values_clean, ci_high * 100))

    # --- Confusion matrix
    if figure_path is not None:
        plot_weighted_conf_matrix(tp_soft, fp_soft, fn_soft, tn_soft, figure_path=figure_path)

    return results
    
def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, ci=0.95):
    '''
    Helper function to obtain the CI of a metric
    Input: 
        - target and pred values
        - metric to be computed
        - number of bootstrap and confidence interval
    Out: 
        - upper and lower CI values
    '''

    metrics = []
    n = len(y_true)
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    for _ in range(n_boot):
        idxs = np.random.choice(n, n, replace=True)
        y_true_sample = y_true[idxs]
        y_pred_sample = y_pred[idxs]
        metrics.append(metric_fn(y_true_sample, y_pred_sample))

    lower = np.percentile(metrics, (1 - ci) / 2 * 100)
    upper = np.percentile(metrics, (1 + ci) / 2 * 100)
    return lower, upper

def extract_arc_from_report(report_file: str) -> str | None:
    '''
    Helper fucntion to extract the architecture name of a report that may be broken 
    bc of inconsistencies with metrics. Used to automize model loading
    Input: 
        - json report file
    Output: 
        - name of the architecture
    '''
    
    with open(report_file, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    try:
        obj = json.loads(txt)
        return obj.get("arguments", {}).get("arc")
    except json.JSONDecodeError:
        pass

    opens = txt.count("{")
    closes = txt.count("}")
    if closes < opens:
        txt_fixed = txt + ("}" * (opens - closes))
        try:
            obj = json.loads(txt_fixed)
            return obj.get("arguments", {}).get("arc")
        except json.JSONDecodeError:
            pass

    m = re.search(r'"arc"\s*:\s*"([^"]+)"', txt)
    if m:
        return m.group(1)

    return None

def ensemble_with_rules(reg_preds, scaler, threshold_high=300, threshold_low=200, slope_thresh=0.3, difference=False, x_last=None):
    
    '''
    Helper fucntion to eperform ensemble of classification head and predefined rules
    Input: 
        - prediceted values for each 
    Output: 
        - name of the architecture
    '''
    N = reg_preds.shape[0]
    steps = reg_preds.shape[1]
    
    if reg_preds.dim() == 3:
        vals = reg_preds.shape[2]
        y_reshape = reg_preds.reshape(-1, vals)[:, 0]  # flatten for scaler
    else:
        vals = 1
        y_reshape = reg_preds.reshape(-1, 1)

    # Add x_last if difference=True
    if difference:
        # Expand x_last to match flattened shape
        x_last_expanded = x_last.repeat(1, steps).reshape(-1, 1)
        # Add only to the first value if vals > 1
        if vals == 1:
            y_reshape = y_reshape + x_last_expanded
        else:
            y_reshape[:, 0] = y_reshape[:, 0] + x_last_expanded[:, 0]

    y_inv = scaler.inverse_transform(y_reshape)
    y_inv=y_inv.reshape(N, steps)
    
    labels = np.zeros(N, dtype=np.int64)
    '''
    for i in range(N):
        preds = y_inv[i]
        last_val = preds[-1]
        slope = preds[-1] - preds[-2]

        if last_val > threshold_high:
            labels[i] = 1
        elif last_val < threshold_low:
            labels[i] = 0
        elif slope > slope_thresh:
            labels[i] = 1
        elif slope < -slope_thresh:
            labels[i] = 0
        else:
            # copiar etiqueta de instancia anterior
            if i > 0:
                labels[i] = labels[i-1]
            else:
                labels[i] = 0  # o 1 si quieres iniciar con 1
    '''
    for i in range(N):
        preds = y_inv[i]
        last_val = preds[-1]
        if last_val >= (threshold_high*0.9): 
            labels[i] = 1
        else:
            labels[i] = 0
            
    return labels

def smooth_binary_predictions(
        y_pred_cl,
        threshold_up=0.6,
        threshold_low=0.3,
        trend_len=2,
        upslope_rel=0.5,
        mode='bidirectional_slope'
    ):

    y_pred_cl = np.asarray(y_pred_cl)
    n = len(y_pred_cl)
    y_bin = np.zeros(n, dtype=int)

    # --- inicialización correcta: clase más cercana ---
    y_bin[0] = 1 if y_pred_cl[0] >= 0.5 else 0

    # ================================================================
    #   MODE 2 — BIDIRECTIONAL_SLOPE
    # ================================================================
    if mode == 'bidirectional_slope':

        for i in range(1, n):
            val = y_pred_cl[i]

            # Hard thresholds
            if val >= threshold_up:
                y_bin[i] = 1
                continue

            if val <= threshold_low:
                y_bin[i] = 0
                continue

            y_bin[i] = y_bin[i-1]

            if i >= trend_len:
                # segment of last trend_len+1 points
                segment = y_pred_cl[i-trend_len:i+1]
                deltas = np.diff(segment)

                # all positive slopes → strong upward trend
                if np.all(deltas > 0):
                    y_bin[i] = 1
                # all negative slopes → strong downward trend
                elif np.all(deltas < 0):
                    y_bin[i] = 0


    # ================================================================
    #   MODE 3 — UP_SLOPE (solo activa 1, nunca apaga)
    # ================================================================
    elif mode == 'up_slope':

        for i in range(1, n):
            val = y_pred_cl[i]

            # Si ≥ threshold_up ⇒ instantáneo a 1
            if val >= threshold_up:
                y_bin[i] = 1
                continue

            # keep previous state
            y_bin[i] = y_bin[i-1]

            if i >= trend_len:
                segment = y_pred_cl[i-trend_len:i+1]
                
                eps = 1e-6
                rel_deltas = []
                for k in range(1, len(segment)):
                    prev = max(segment[k-1], eps)
                    rel_deltas.append((segment[k] - segment[k-1]) / prev)

                rel_deltas = np.array(rel_deltas) 

                # Subida fuerte: todos los deltas relativos > upslope_rel
                strong_up = np.all(rel_deltas > 0) and (y_pred_cl[i] - y_pred_cl[i-trend_len] > upslope_rel)

                # Bajada fuerte: todos los deltas relativos < -upslope_rel
                strong_down = np.all(rel_deltas < 0) and (y_pred_cl[i] - y_pred_cl[i-trend_len] < - upslope_rel)
                if strong_up:
                    y_bin[i] = 1
                elif strong_down:
                    y_bin[i] = 0

    elif mode == 'hysteresis':

        for i in range(1, n):

            val = y_pred_cl[i]
            
            # --- caso 1: por encima de threshold_up ---
            if val >= threshold_up:
                y_bin[i] = 1
                continue
            
            # --- caso 2: por debajo de threshold_low ---
            elif val <= threshold_low:
                y_bin[i] = 0
                continue
            
            if i >= trend_len:
                segment = y_pred_cl[i-trend_len+1:i+1]
                delta=np.diff(segment)
                # all positive slopes → strong upward trend
                if np.all(segment> threshold_low): #:and np.all(delta>0):
                    y_bin[i] = 1
                # all negative slopes → strong downward trend
                elif np.all(segment < threshold_up): # and np.all(delta<0):
                    y_bin[i] = 0

            else:
                y_bin[i] = y_bin[i-1]   # NO CAMBIA
            

        
    elif mode == 'history':
        for i in range(1, n):
            val = y_pred_cl[i]

            # Si ≥ threshold_up ⇒ instantáneo a 1
            if val >= threshold_up:
                y_bin[i] = 1
                continue

            # keep previous state
            y_bin[i] = y_bin[i-1]
            
            if i >= trend_len:
                segment = y_pred_cl[i-trend_len:i+1]
                
                eps = 1e-6
                rel_deltas = []
                for k in range(1, len(segment)):
                    prev = max(segment[k-1], eps)
                    rel_deltas.append((segment[k] - segment[k-1]) / prev)

                rel_deltas = np.array(rel_deltas) 

                # Subida fuerte: todos los deltas relativos > upslope_rel
                strong_up = np.all(rel_deltas > 0) # and (y_pred_cl[i] - y_pred_cl[i-trend_len] > upslope_rel)

                # Bajada fuerte: todos los deltas relativos < -upslope_rel
                strong_down = np.all(rel_deltas < 0) and (y_pred_cl[i] - y_pred_cl[i-trend_len] < - upslope_rel)
                if strong_up:
                    y_bin[i] = 1
                elif strong_down:
                    y_bin[i] = 0
                    
    elif mode == 'fixed':
        for i in range(1, n):
            val = y_pred_cl[i]

            # Si ≥ threshold_up ⇒ instantáneo a 1
            if val >= threshold_up:
                y_bin[i] = 1
            else: 
                y_bin[i] = 0

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return y_bin

def grid_search_smoothing(y_true, y_pred_cl):

    #modes = ["bidirectional_slope", "up_slope", "hysteresis", "history"]
    modes = ["hysteresis"]
    thr_up_vals  = [0.7]
    thr_low_vals = [0.3]
    trend_vals   = [3]
    slope_vals   = [0.05]
    all_results = []
    best_global = {"mode": None, "params": None, "score": np.inf, "y_bin": None}

    for mode in modes:
        for th_up in thr_up_vals:
            for th_low in thr_low_vals:
                for tl in trend_vals:
                    for slope in slope_vals:

                        y_bin = smooth_binary_predictions(
                            y_pred_cl,
                            threshold_up=th_up,
                            threshold_low=th_low,
                            trend_len=tl,
                            upslope_rel=slope,
                            mode=mode
                        )

                        tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()
                        score = fp + 10 * fn

                        all_results.append([mode, th_up, th_low, tl, slope, fp, fn, score])
                        print(f"mode {mode}, th_up {th_up}, th_low {th_low}, tl, {tl}, slope {slope}, fp {fp}, fn {fn}")
                        if score < best_global["score"]:
                            #print(f"mode {mode}, th_up {th_up}, th_low {th_low}, tl, {tl}, slope {slope}, fp {fp}, fn {fn}")
                                                
                            best_global = {
                                "mode": mode,
                                "params": (th_up, th_low, tl, slope),
                                "score": score,
                                "y_bin": y_bin
                            }

    results_df = pd.DataFrame(
        all_results,
        columns=["mode","thr_up","thr_low","trend_len","slope","FP","FN","score"]
    )

    return (
        best_global["mode"],
        *best_global["params"]
        #best_global["y_bin"],
        #results_df
    )
    
    
def comparison_to_rule(filepath, model, device, scaler, threshold, lookback=12, forecast=6, slope=0.3, thr_up=0.7, thr_down=0.3, trend=2, slope_rel=1, mode="default", figure_path=None):
    print("PERFORMING COMPARISON RULE AND AI SYSTEMS")
    
    
    (X_train, y_train, y_train_bin, w_cl_train, w_reg_train), (X_val, y_val, y_val_bin, w_cl_val, w_reg_val), (X_test, y_test, y_test_bin, w_cl_test, w_reg_test), scaler, scaler_y, train_std = preprocess_data(filepath, lookback, forecast, threshold)


    train_loader, val_loader, test_loader = create_dataloaders(X_train, y_train, y_train_bin, w_cl_train, w_reg_train, X_val, y_val, y_val_bin, w_cl_val, w_reg_val, X_test, y_test, y_test_bin, w_cl_test, w_reg_test,
                                                               scaler=scaler, train_std=train_std, batch_size=1, dataaug=False, shuffle=False)



    # ---- Función auxiliar para obtener predicciones y valores reales de un loader ----
    def get_preds_and_labels(loader):
        y_list, yreg_list, preds_list = [], [], []
        model.eval()
        with torch.no_grad():
            for Xb, yb, yb_bin, _, _ in loader:
                Xb = Xb.to(device)
                out = model(Xb)
                if isinstance(out, tuple):
                    _, logits = out
                else:
                    logits = out
                p = torch.sigmoid(logits).cpu().numpy().ravel()
                preds_list.append(p)
                y_list.append(yb_bin.cpu().numpy())
                yreg_list.append(yb.cpu().numpy())
        y_bin_real = np.concatenate(y_list).ravel()
        y_reg_all=np.concatenate(yreg_list)
        preds_all = np.concatenate(preds_list)
        y_bin_pred = smooth_binary_predictions(preds_all, thr_up, thr_down, trend, slope_rel, mode)

        return preds_all, y_reg_all, y_bin_pred, y_bin_real

    # ---- Obtener predicciones y etiquetas de train, val y test ----
    y_cl_train, yreg_train, y_bin_train, y_bin_train_real = get_preds_and_labels(train_loader)
    print("num elements train", len(yreg_train))
    y_cl_val, yreg_val, y_bin_val , y_bin_val_real = get_preds_and_labels(val_loader)
    print("num elements val", len(yreg_val))
    y_cl_test, yreg_test, y_bin_test, y_bin_test_real  = get_preds_and_labels(test_loader)
    print("num elements test", len(yreg_test))

    # ---- Concatenar los tres conjuntos ----
    y_cl_all = np.concatenate([y_cl_train, y_cl_val, y_cl_test], axis=0)
    y_reg_all = np.concatenate([yreg_train, yreg_val, yreg_test], axis=0)
    y_bin_all = np.concatenate([y_bin_train, y_bin_val, y_bin_test], axis=0)
    y_real_bin_all = np.concatenate([y_bin_train_real ,y_bin_val_real , y_bin_test_real ], axis=0)
    
    print(len(y_reg_all), len(y_bin_all), len(y_real_bin_all))

    # ---- A partir de aquí, usamos el mismo código que tu función original ----
    N, steps = y_reg_all.shape
    y_reshape = y_reg_all.reshape(-1, 1)
    y_inv = scaler_y.inverse_transform(y_reshape).reshape(N, steps)

    y_bin_AI = y_bin_all.copy()
    y_bin_AI_real = y_real_bin_all.copy()
    y_bin_RS = np.zeros_like(y_bin_AI)

    # Cálculo de derivada aproximada
    diff = np.zeros_like(y_inv[:, 0])
    diff[1:] = (y_inv[1:, 0] - y_inv[:-1, 0]) / 10.0

    # Sistema de reglas
    for i in range(1, len(y_inv)):
        if (y_inv[i, 0] > threshold or 
            (y_inv[i, 0] > (2/3) * threshold and diff[i] > slope)):
            y_bin_RS[i] = 1
        elif y_inv[i, 0] > (2/3) * threshold and diff[i] > -slope:
            y_bin_RS[i] = y_bin_RS[i-1]

    above_thr = np.sum(y_inv[:, 0] > threshold)
    transitions = np.sum((y_inv[1:, 0] == 1) & (y_inv[:-1, 0] == 0))
    sp = above_thr + 2 * transitions

    # Costes
    cost_RS = np.sum(y_bin_RS) / len(y_bin_RS)
    cost_AI = np.sum(y_bin_AI) / len(y_bin_AI)
    cost_real_IA = np.sum(y_bin_AI_real) / len(y_bin_AI_real)
    cost_min = sp / len(y_bin_AI)


    unsafe_RS = np.where(
        (y_bin_RS == 0)
        & (np.any(y_inv[:, 0:3] > threshold, axis=1))
    )[0]

    unsafe_AI = np.where(
        (y_bin_AI == 0)
        & (np.any(y_inv[:, 0:3] > threshold, axis=1))
    )[0]

    unsafe_AI_real = np.where(
        (y_bin_AI_real == 0)
        & (np.any(y_inv[:, 0:3] > threshold, axis=1))
    )[0]
    

    safety_RS = 1 - len(unsafe_RS) / len(y_reg_all)
    safety_AI = 1 - len(unsafe_AI) / len(y_reg_all)
    safety_AI_real = 1 - len(unsafe_AI_real) / len(y_reg_all)

    # Transiciones
    trans_RS = np.where((y_bin_RS[1:] == 1) & (y_bin_RS[:-1] == 0))[0] + 1
    trans_AI = np.where((y_bin_AI[1:] == 1) & (y_bin_AI[:-1] == 0))[0] + 1
    trans_AI_real = np.where((y_bin_AI_real[1:] == 1) & (y_bin_AI_real[:-1] == 0))[0] + 1

    val_RS = y_inv[trans_RS - 1, 0] if len(trans_RS) > 0 else []
    val_AI = y_inv[trans_AI - 1, 0] if len(trans_AI) > 0 else []
    val_AI_real = y_inv[trans_AI_real - 1, 0] if len(trans_AI_real) > 0 else []

    # Prints
    print(f"Coste RS: {cost_RS:.4f}, Coste AI: {cost_AI:.4f}, Coste AI real : {cost_real_IA:.4f}, Coste min: {cost_min:.4f}")
    print(f"Seguridad RS: {safety_RS:.4f}, Seguridad AI: {safety_AI:.4f}, Seguridad AI real: {safety_AI_real:.4f}; Seguridad max: 1.00")
    print(f"Violaciones RS: {len(unsafe_RS)}, Violaciones AI: {len(unsafe_AI)}, Violaciones AI real: {len(unsafe_AI_real)}")
    print(f"Transiciones RS: {trans_RS.tolist()}")
    print(f"Transiciones AI: {trans_AI.tolist()}")
    print(f"Transiciones AI real: {trans_AI_real.tolist()}")
    #print(f"Valores activadores RS: {val_RS}")
    #print(f"Valores activadores AI: {val_AI}")
    #print(f"Valores activadores AI real: {val_AI_real}")
    print(f"Indices UNSAFE_RS ({len(unsafe_RS)} casos): {unsafe_RS.tolist()}")
    print(f"Indices UNSAFE_AI ({len(unsafe_AI)} casos): {unsafe_AI.tolist()}")
    print(f"Indices UNSAFE_AI_REAL ({len(unsafe_AI_real)} casos): {unsafe_AI_real.tolist()}")
    
    print("Valores y_cl_all en UNSAFE_AI:", y_cl_all[unsafe_AI].tolist())

    # Figura
    plt.figure(figsize=(6, 6))
    '''
    plt.scatter(cost_RS, safety_RS, color='orange', label='Rule System')
    plt.scatter(cost_AI, safety_AI, color='lightblue', label='AI Model')
    plt.scatter(cost_real_IA, safety_AI_real, color='blue', label='True AI Model')
    plt.scatter(cost_min, 1, color='green', label='Best Model')
    '''
    plt.scatter(cost_RS, len(unsafe_RS), color='orange', label='Rule System')
    plt.scatter(cost_AI, len(unsafe_AI), color='lightblue', label='AI Model')
    plt.scatter(cost_real_IA, len(unsafe_AI_real), color='royalblue', label='True AI Model')
    plt.scatter(cost_min, 0, color='yellowgreen', label='Best Model')
    plt.xlabel('Cost')
    plt.ylabel('Security violations')
    plt.legend()
    plt.savefig(figure_path)
    plt.close()
    

    return {
        "cost_AI": cost_AI,
        "cost_RS": cost_RS,
        "safety_AI": safety_AI,
        "safety_RS": safety_RS
    }
