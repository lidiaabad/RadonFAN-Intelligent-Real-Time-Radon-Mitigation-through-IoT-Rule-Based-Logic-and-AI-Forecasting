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
        self.q_l = quant_lower # low quantile
        self.q_h = quant_higher # high quantile

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
        self.low_frac = low_frac 
        self.high_frac = high_frac  
        self.w = weight_factor  # extra weight

    def forward(self, y_pred, y_true):
        error = torch.abs(y_true - y_pred)
        
        y_min, y_max = torch.min(y_true), torch.max(y_true)
        low_thresh = y_min + self.low_frac * (y_max - y_min)
        high_thresh = y_min + self.high_frac * (y_max - y_min)

        overestimate_low = (y_true < low_thresh) & (y_pred > y_true)
        underestimate_high = (y_true > high_thresh) & (y_pred < y_true)
        
        weight = torch.ones_like(y_true)
        weight = torch.where(overestimate_low | underestimate_high, self.w, weight)

        weighted_error = error * weight
        return torch.mean(weighted_error)
    

class DenseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets, weights=None):
        loss = (predictions - targets) ** 2 # weighted MSE

        if weights is not None:
            weights = weights.view(-1, 1).to(loss.device)
            loss = loss * weights
        return loss.mean()

class DenseLossPersistence(nn.Module):
    def __init__(self, lambda_p=0.1, threshold=None):
        """
        lambda_p: persistene term 
        threshold: threshold for penalizing if close enough to last value
        """
        super().__init__()
        self.lambda_p = lambda_p
        self.threshold = threshold

    def forward(self, predictions, targets, x_last, weights=None):

        # --- MSE base ---
        loss = (predictions - targets) ** 2

        if weights is not None:
            weights = weights.view(-1, 1, 1).to(loss.device)
            loss = loss * weights

        mse_loss = loss.mean()

        if x_last.ndim == 2:
            x_last = x_last.unsqueeze(1)
            
        x_last = x_last.view(x_last.size(0), -1)  # -> [B, 1]
        x_last_expanded = x_last.expand(-1, predictions.size(1))  # -> [B, T]
        diff = torch.norm(predictions - x_last_expanded, dim=-1)  # [B, T]

        if self.threshold is not None:
            persist_penalty = torch.relu(self.threshold - diff).mean()
        else:
            persist_penalty = -diff.mean()

        total_loss = mse_loss + self.lambda_p * persist_penalty
        return total_loss

class DenseBandLoss(nn.Module):
    def __init__(self, band=(200, 350), over_w=2.0, under_w=2.0, scaler=None):

        super().__init__()
        self.low, self.high = band #near threhsold band 
        self.over_w = over_w # weight for overestimation 
        self.under_w = under_w #weight for underestimation
        self.scaler = scaler # for value computation

    def forward(self, predictions, targets, weights=None):
        
        device = predictions.device

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


class HingeLossWithPosWeight(nn.Module):
    def __init__(self, pos_weight=1.0, margin=1.0):

        super().__init__()
        self.pos_weight = pos_weight
        self.margin = margin # hinge margin

    def forward(self, logits, targets):

        targets = targets.float()
        y = targets * 2 - 1  # map 0→-1, 1→+1

        # hinge loss per sample
        losses = torch.clamp(self.margin - y * logits, min=0)

        weights = torch.where(targets == 1, self.pos_weight, 1.0)
        losses = losses * weights

        return losses.mean()


class SkillLoss(nn.Module):
    def __init__(self, margin=0.0, eps=1e-6):
        super().__init__()
        self.margin = margin  # margin to baseline improvement
        self.eps = eps 

    def forward(self, y_pred, y_true, y_baseline):

        mse_model = F.mse_loss(y_pred, y_true, reduction="mean")
        mse_baseline = F.mse_loss(y_baseline, y_true, reduction="mean")

        ratio = mse_model / (mse_baseline + self.eps)

        loss = torch.clamp(ratio - self.margin, min=0.0)
        return loss


class AsymmetricMSELoss(nn.Module):
    def __init__(self, alpha=3.0, beta=1.0):
        super().__init__()
        self.alpha = alpha # for underestimation
        self.beta = beta # for overestimation

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        weights = torch.where(error > 0, self.alpha, self.beta)
        loss = weights * (error ** 2)

        return loss.mean()


class AsymmetricPersistenceMSELoss(nn.Module):
    def __init__(self, alpha=3.0, beta=1.0, gamma=0.1):

        super().__init__()
        self.alpha = alpha # for underestimation
        self.beta = beta # for overestimation
        self.gamma = gamma # for persistence

    def forward(self, y_pred, y_true, last_X_val):
        
        error = y_true - y_pred
        weights = torch.where(error > 0, self.alpha, self.beta)  
        loss_asym = weights * (error ** 2)

        persist_error = y_pred - last_X_val
        loss_persist = self.gamma * (persist_error ** 2)

        loss = loss_asym - loss_persist
        
        return loss.mean()


def compute_loss_per_class(logits, targets, pos_weight=None, weights=None):
    '''
    Function to compute loss per class without weights so that the weighted loss effect ccan be analyze in the log file
    Input: 
        - logits
        - targets
        - pos_weight (w_p)
        - weights (w_i)
    Output: 
        - positive class loss
        - negative class loss
    '''

    if pos_weight is not None:
        if isinstance(pos_weight, float) or (torch.is_tensor(pos_weight) and pos_weight.numel() == 1):
            pos_weight_tensor = torch.tensor([float(pos_weight)], device=logits.device)
        else:
            raise ValueError("pos_weight debe ser un escalar para un solo logit")

    if weights is not None:
        weights_tensor = weights.view(-1, 1)

    losses = F.binary_cross_entropy_with_logits(logits.float(),targets.float(),weight=weights_tensor,pos_weight=pos_weight_tensor,reduction="none")
    
    pos_mask = (targets == 1).squeeze()
    neg_mask = (targets == 0).squeeze()
    pos_loss = losses[pos_mask].mean().item() if pos_mask.any() else 0.0
    neg_loss = losses[neg_mask].mean().item() if neg_mask.any() else 0.0

    return pos_loss, neg_loss
