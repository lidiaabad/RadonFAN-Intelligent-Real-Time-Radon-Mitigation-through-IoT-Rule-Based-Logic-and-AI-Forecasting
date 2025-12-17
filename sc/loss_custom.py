import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        focal_loss=focal_loss.mean()
        return focal_loss
    

class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1).float()

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
        loss = 1 - dice
        return loss
    
class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = y_pred - y_true
        return torch.mean(torch.log(torch.cosh(loss + 1e-12)))
    
class QuantileLoss(nn.Module):
    def __init__(self, quantile):
        super().__init__()
        if not 0 < quantile < 1:
            raise ValueError("Quantile must be between 0 and 1.")
        self.q = quantile

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        loss = torch.max(self.q * error, (self.q - 1) * error)
        return torch.mean(loss)
    
class MSLE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return torch.mean((torch.log1p(y_true) - torch.log1p(y_pred)) ** 2)


class DoubleQuantileLoss(nn.Module):
    def __init__(self, quant_lower, quant_higher):
        super().__init__()
        if not 0 < quant_lower < 1 or not 0 < quant_higher < 1:
            raise ValueError("Quantile must be between 0 and 1.")
        self.q_l = quant_lower
        self.q_h = quant_higher

    def forward(self, y_pred, y_true):
        error_l = y_pred - y_true
        loss_l = torch.max(self.q_l * error_l, (self.q_l - 1) * error_l)
        error_h = y_true - y_pred #reverse
        loss_h = torch.max(self.q_h * error_h, (self.q_h - 1) * error_h)
        loss=loss_l+loss_h
        return torch.mean(loss)

class DynamicExtremeWeightedLoss(torch.nn.Module):
    def __init__(self, low_frac=0.1, high_frac=0.9, weight_factor=2.0):
        super().__init__()
        self.low_frac = low_frac  # Fracción para definir "low"
        self.high_frac = high_frac  # Fracción para definir "high"
        self.w = weight_factor  # Peso extra

    def forward(self, y_pred, y_true):
        error = torch.abs(y_true - y_pred)

        # Calcular extremos dinámicamente en base a min/max del batch
        y_min, y_max = torch.min(y_true), torch.max(y_true)
        low_thresh = y_min + self.low_frac * (y_max - y_min)
        high_thresh = y_min + self.high_frac * (y_max - y_min)

        # Máscara para errores relevantes
        overestimate_low = (y_true < low_thresh) & (y_pred > y_true)
        underestimate_high = (y_true > high_thresh) & (y_pred < y_true)

        # Aplicar pesos
        weight = torch.ones_like(y_true)
        weight = torch.where(overestimate_low | underestimate_high, self.w, weight)

        weighted_error = error * weight
        return torch.mean(weighted_error)
    
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, weights=None):
        loss = (inputs - targets) ** 2
        if weights is not None:
            loss = loss * weights
        return loss.mean()

class DenseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets, weights=None):
        loss = (predictions - targets) ** 2

        if weights is not None:
            weights = weights.view(-1, 1).to(loss.device)
            loss = loss * weights

        return loss.mean()

class DenseLossPersistence(nn.Module):
    def __init__(self, lambda_p=0.1, threshold=None):
        """
        lambda_p: peso del término de persistencia
        threshold: si no es None, penaliza solo si predicciones están muy cerca del último valor
        """
        super().__init__()
        self.lambda_p = lambda_p
        self.threshold = threshold

    def forward(self, predictions, targets, x_last, weights=None):
        """
        predictions: [B, T, D]
        targets:     [B, T, D]
        x_last:      [B, D] o [B, 1, D]
        weights:     [B] opcional
        """
        # --- MSE base ---
        loss = (predictions - targets) ** 2

        if weights is not None:
            weights = weights.view(-1, 1, 1).to(loss.device)
            loss = loss * weights

        mse_loss = loss.mean()

        # --- Penalización de persistencia ---
        if x_last.ndim == 2:
            x_last = x_last.unsqueeze(1)
            
        x_last = x_last.view(x_last.size(0), -1)  # -> [B, 1]
        x_last_expanded = x_last.expand(-1, predictions.size(1))  # -> [B, T]
        diff = torch.norm(predictions - x_last_expanded, dim=-1)  # [B, T]

        if self.threshold is not None:
            # Penaliza solo si está demasiado cerca del último valor
            persist_penalty = torch.relu(self.threshold - diff).mean()
        else:
            # Penaliza proporcionalmente a la similitud (negativo = cerca)
            persist_penalty = -diff.mean()

        total_loss = mse_loss + self.lambda_p * persist_penalty
        return total_loss


class DenseLossProb(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets, weights=None):
        mu = predictions[..., 0].squeeze(-1)
        log_sigma = predictions[..., 1].squeeze(-1)
        sigma = torch.exp(log_sigma).squeeze(-1)
        #print(targets.shape, mu.shape, log_sigma.shape, sigma.shape, predictions.shape)

        # NLL Gussiana
        loss = 0.5 * ((targets - mu)**2 / (sigma**2 + 1e-8) + 2 * log_sigma)

        if weights is not None:
            weights = weights.view(-1, 1).to(loss.device)
            loss = loss * weights

        return loss.mean()


class WeightedLagDenseLoss(nn.Module):
    def __init__(self, lag_penalty=0.5, max_lag=None, decay="exp", reduction="mean", k=0.4):
        """
        lag_penalty: peso global de penalización
        max_lag: máximo desfase (por defecto = T-1)
        decay: 'exp', 'linear' o 'inv'
        k: parámetro de decaimiento para 'exp'
        """
        super().__init__()
        self.lag_penalty = lag_penalty
        self.max_lag = max_lag
        self.decay = decay
        self.reduction = reduction
        self.k = k

    def forward(self, predictions, targets, weights=None):

        B, T = predictions.shape
        max_lag = self.max_lag or (T - 1)
        device = predictions.device

        mse = (predictions - targets) ** 2

        w = 1.0 / torch.arange(1, max_lag + 1, device=device).float()

        lag_losses = []
        for lag in range(1, max_lag + 1):
            shifted_targets = torch.roll(targets, shifts=-lag, dims=1)
            shifted_targets[:, -lag:] = targets[:, -1].unsqueeze(1)
            lag_mse = (predictions - shifted_targets) ** 2
            lag_losses.append(w[lag - 1] * lag_mse)

        lag_loss = torch.stack(lag_losses, dim=0).sum(dim=0)
        total_loss = mse + self.lag_penalty * lag_loss

        if weights is not None:
            weights = weights.view(-1, 1).to(device)
            total_loss = total_loss * weights

        return total_loss.mean()

class DenseBandLoss(nn.Module):
    def __init__(self, band=(200, 350), over_w=2.0, under_w=2.0, scaler=None):

        super().__init__()
        self.low, self.high = band
        self.over_w = over_w
        self.under_w = under_w
        self.scaler = scaler

    def forward(self, predictions, targets, weights=None):
        """
        predictions, targets: (batch, horizons)
        weights: optional (batch,) sample weights
        """
        device = predictions.device

        # Optionally inverse-transform to original scale
        B, H = predictions.shape
        preds_orig = self.scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1))
        targs_orig = self.scaler.inverse_transform(targets.detach().cpu().numpy().reshape(-1, 1))
        preds_orig = torch.tensor(preds_orig.reshape(B, H), device=device, dtype=predictions.dtype)
        targs_orig = torch.tensor(targs_orig.reshape(B, H), device=device, dtype=targets.dtype)


        # Identify values in the band
        in_band = (targs_orig >= self.low) & (targs_orig <= self.high)

        # Base weights
        w = torch.ones_like(predictions)
        w = torch.where(in_band & (preds_orig > targs_orig), self.over_w, w) #overstimation
        w = torch.where(in_band & (preds_orig < targs_orig), self.under_w, w) #understimation

        loss = (predictions - targets) ** 2
        loss = loss * w

        # Apply optional sample-level weights
        if weights is not None:
            weights = weights.view(-1, 1).to(device)
            loss = loss * weights

        return loss.mean()

class WeightedTemporalMSELoss(nn.Module):
    def __init__(self, horizon_weights=None, mode="linear"):
        """
        horizon_weights: list/torch.Tensor with manual weights for each horizon
        mode: 'linear', 'exp', or 'custom'
        """
        super().__init__()
        self.horizon_weights = horizon_weights
        self.mode = mode

    def forward(self, y_pred, y_true):
        """
        y_pred, y_true: shape [B, forecast_len]
        """
        forecast_len = y_true.shape[1]
        device = y_true.device

        # Create weights
        if self.horizon_weights is None:
            weights = torch.linspace(1.0, 2.0, steps=forecast_len, device=device)
            #weights = torch.exp(torch.linspace(0, 1, steps=forecast_len, device=device))
        else:
            weights = torch.tensor(self.horizon_weights, dtype=torch.float32, device=device)

        # Weighted MSE
        sq_error = (y_pred - y_true) ** 2                    # [B, L]
        weighted_sq_error = sq_error * weights.unsqueeze(0)  # broadcast weights
        return weighted_sq_error.mean()

class HingeLossWithPosWeight(nn.Module):
    def __init__(self, pos_weight=1.0, margin=1.0, reduction='mean'):
        """
        pos_weight: scalar > 0, weight applied to positive-class samples
        margin: hinge margin (default 1.0)
        reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.margin = margin
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: raw model outputs (N,)
        targets: binary labels 0 or 1 (N,)
        """
        targets = targets.float()
        y = targets * 2 - 1  # map 0→-1, 1→+1

        # hinge loss per sample
        losses = torch.clamp(self.margin - y * logits, min=0)

        # weight positives
        weights = torch.where(targets == 1, self.pos_weight, 1.0)
        losses = losses * weights

        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        return losses

class GradNorm(nn.Module):
    def __init__(self, shared_params, n_tasks=2, alpha=1.5, device="cpu"):
        """
        GradNorm for multi-task loss balancing.
        
        Args:
            shared_params (iterable): Parameters shared between tasks (used to compute grad norms).
            n_tasks (int): Number of tasks/losses.
            alpha (float): GradNorm alpha hyperparameter (controls aggressiveness).
            device (str): "cpu" or "cuda".
        """
        super().__init__()
        self.shared_params = list(shared_params)
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.device = device
        
        # Trainable loss weights
        self.loss_weights = nn.Parameter(torch.ones(n_tasks, device=device))
        
        # Store initial losses for relative training rate computation
        self.initial_losses = None

    def compute_grad_norm(self, loss, retain_graph=False):
        """Compute average gradient norm of loss w.r.t shared parameters."""
        grads = torch.autograd.grad(loss, self.shared_params,
                                    retain_graph=retain_graph,
                                    create_graph=True)
        grad_norms = [g.norm() for g in grads if g is not None]
        return torch.stack(grad_norms).mean()

    def forward(self, losses, epoch):

        assert len(losses) == self.n_tasks, "Number of losses must match n_tasks."

        # Weighted losses
        weighted_losses = [self.loss_weights[i] * losses[i] for i in range(self.n_tasks)]
        combined_loss = sum(weighted_losses)
        
        # Store initial losses at first epoch
        if self.initial_losses is None and epoch == 1:
            self.initial_losses = torch.tensor([l.item() for l in losses], device=self.device)

        # Compute grad norms for each weighted loss
        grad_norms = [self.compute_grad_norm(weighted_losses[i], retain_graph=True)
                      for i in range(self.n_tasks)]
        grad_norms = torch.stack(grad_norms)

        # Compute relative inverse training rates
        current_losses = torch.tensor([l.item() for l in losses], device=self.device)
        loss_ratios = current_losses / (self.initial_losses + 1e-8)
        inverse_train_rates = loss_ratios / loss_ratios.mean()

        # Compute target grad norms
        avg_grad_norm = grad_norms.mean()
        target_grad_norms = avg_grad_norm * (inverse_train_rates ** self.alpha)

        # GradNorm loss
        gradnorm_loss = F.l1_loss(grad_norms, target_grad_norms.detach())

        return combined_loss, gradnorm_loss

    def normalize_weights(self):
        
        with torch.no_grad():
            self.loss_weights.data = self.loss_weights.data / self.loss_weights.sum()


class SkillLoss(nn.Module):
    def __init__(self, margin=0.0, eps=1e-6):
        """
        margin: how much better than baseline the model must be (0 = at least equal).
        eps: numerical stability.
        """
        super().__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, y_pred, y_true, y_baseline):
        """
        y_pred: model predictions, shape (batch, horizon)
        y_true: ground truth, same shape
        y_baseline: baseline forecast (e.g. persistence), same shape
        """
        mse_model = F.mse_loss(y_pred, y_true, reduction="mean")
        mse_baseline = F.mse_loss(y_baseline, y_true, reduction="mean")

        # ratio: relative to baseline
        ratio = mse_model / (mse_baseline + self.eps)

        # apply margin
        loss = torch.clamp(ratio - self.margin, min=0.0)
        return loss


class AsymmetricMSELoss(nn.Module):
    def __init__(self, alpha=3.0, beta=1.0, reduction='mean'):
        """
        alpha: peso cuando la predicción está por debajo del valor real (penaliza rezago).
        beta:  peso cuando la predicción está por encima del valor real.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        weights = torch.where(error > 0, self.alpha, self.beta)  # >0 => pred < real
        loss = weights * (error ** 2)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class AsymmetricPersistenceMSELoss(nn.Module):
    def __init__(self, alpha=3.0, beta=1.0, gamma=0.1, reduction='mean'):
        """
        alpha: peso cuando la predicción está por debajo del valor real (penaliza rezago)
        beta:  peso cuando la predicción está por encima del valor real
        gamma: peso extra para penalizar persistencia (predicción ≈ last_X_val)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_pred, y_true, last_X_val):
        # Error asimétrico clásico
        error = y_true - y_pred
        weights = torch.where(error > 0, self.alpha, self.beta)  # >0 => pred < real
        loss_asym = weights * (error ** 2)

        # Penalización de persistencia
        persist_error = y_pred - last_X_val
        loss_persist = self.gamma * (persist_error ** 2)

        # Loss total
        loss = loss_asym - loss_persist
        
        return loss.mean()


def compute_loss_per_class(logits, targets, pos_weight=None, weights=None):

    if pos_weight is not None:
        if isinstance(pos_weight, float) or (torch.is_tensor(pos_weight) and pos_weight.numel() == 1):
            pos_weight_tensor = torch.tensor([float(pos_weight)], device=logits.device)
        else:
            raise ValueError("pos_weight debe ser un escalar para un solo logit")
    else:
        pos_weight_tensor = None

    # preparar weights
    if weights is not None:
        # asegurar shape [batch,1] para broadcasting correcto
        weights_tensor = weights.view(-1, 1)
    else:
        weights_tensor = None

    # BCE sin reducción para tener loss por muestra
    losses = F.binary_cross_entropy_with_logits(
        logits.float(),
        targets.float(),
        weight=weights_tensor,
        pos_weight=pos_weight_tensor,
        reduction="none"
    )
    # separar positivos y negativos
    pos_mask = (targets == 1).squeeze()
    neg_mask = (targets == 0).squeeze()

    pos_loss = losses[pos_mask].mean().item() if pos_mask.any() else 0.0
    neg_loss = losses[neg_mask].mean().item() if neg_mask.any() else 0.0

    return pos_loss, neg_loss