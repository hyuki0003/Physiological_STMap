import torch
import torch.nn as nn
class NegPearson(nn.Module):
    def __init__(self):
        super(NegPearson, self).__init__()

    def forward(self, preds, labels):
        """
        preds, labels: [B, T]
        """
        preds = preds - preds.mean(dim=1, keepdim=True)
        labels = labels - labels.mean(dim=1, keepdim=True)

        preds = preds / (preds.std(dim=1, keepdim=True) + 1e-8)
        labels = labels / (labels.std(dim=1, keepdim=True) + 1e-8)

        corr = (preds * labels).mean(dim=1)  # shape: [B]
        loss = 1 - corr.mean()
        return loss

class PSDNegPearsonLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PSDNegPearsonLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        # Compute the FFT of predictions and targets
        pred_fft = torch.fft.fft(pred, dim=-1)
        target_fft = torch.fft.fft(target, dim=-1)

        # Compute the Power Spectral Density (PSD)
        pred_psd = torch.abs(pred_fft) ** 2
        target_psd = torch.abs(target_fft) ** 2

        # Flatten PSD for Pearson correlation calculation
        pred_psd_flat = pred_psd.view(pred_psd.size(0), -1)
        target_psd_flat = target_psd.view(target_psd.size(0), -1)

        # Normalize PSDs by subtracting the mean
        pred_psd_flat = pred_psd_flat - pred_psd_flat.mean(dim=1, keepdim=True)
        target_psd_flat = target_psd_flat - target_psd_flat.mean(dim=1, keepdim=True)

        # Compute norms to prevent division by zero
        pred_psd_norm = torch.norm(pred_psd_flat, p=2, dim=1) + self.epsilon
        target_psd_norm = torch.norm(target_psd_flat, p=2, dim=1) + self.epsilon

        # Compute cosine similarity (dot product divided by norms)
        cosine_similarity = torch.sum(pred_psd_flat * target_psd_flat, dim=1) / (pred_psd_norm * target_psd_norm)

        # Negate the Pearson correlation (1 - correlation)
        neg_pearson_psd_loss = 1 - cosine_similarity.mean()

        return neg_pearson_psd_loss


class PSDEntropyLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PSDEntropyLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        # Compute the FFT of predictions and targets
        pred_fft = torch.fft.fft(pred, dim=-1)
        target_fft = torch.fft.fft(target, dim=-1)

        # Compute the Power Spectral Density (PSD)
        pred_psd = torch.abs(pred_fft) ** 2
        target_psd = torch.abs(target_fft) ** 2

        # Normalize PSDs to prevent division by zero
        pred_psd = pred_psd / (pred_psd.sum(dim=-1, keepdim=True) + self.epsilon)
        target_psd = target_psd / (target_psd.sum(dim=-1, keepdim=True) + self.epsilon)

        # Compute Cross-Entropy between the PSDs
        psd_loss = -torch.sum(target_psd * torch.log(pred_psd + self.epsilon), dim=-1).mean()

        return psd_loss


class CombinedLoss(nn.Module):
    def __init__(self, psd_weight=1, pearson_weight=0.05):
        super(CombinedLoss, self).__init__()
        self.NegPearsonLoss = NegPearson()
        self.PSDEntropyLoss = PSDEntropyLoss()
        self.PSDNegPearsonLoss = PSDNegPearsonLoss()
        self.psd_weight = psd_weight
        self.pearson_weight = pearson_weight

    def forward(self, preds, labels):
        # Pearson Loss
        loss1 = self.NegPearsonLoss(preds, labels)

        # PSD Entropy Loss
        loss2 = self.PSDEntropyLoss(preds, labels)

        loss3 = self.PSDNegPearsonLoss(preds, labels)

        # Combine both losses (weighted sum)
        combined_loss = self.pearson_weight * loss1 + self.psd_weight * loss2 + loss3

        return combined_loss


class PearsonMSELoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        alpha: Pearson 손실 비중 (0~1)
        (1 - alpha): MSE 손실 비중
        """
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, preds, labels):
        """
        preds, labels: [B, T]
        """
        # Pearson loss
        preds_norm = preds - preds.mean(dim=1, keepdim=True)
        labels_norm = labels - labels.mean(dim=1, keepdim=True)

        preds_norm = preds_norm / (preds.std(dim=1, keepdim=True) + 1e-8)
        labels_norm = labels_norm / (labels.std(dim=1, keepdim=True) + 1e-8)

        corr = (preds_norm * labels_norm).mean(dim=1)  # shape: [B]
        pearson_loss = 1 - corr.mean()

        # MSE loss
        mse_loss = self.mse(preds, labels)

        # Combined loss
        total_loss = self.alpha * pearson_loss + (1 - self.alpha) * mse_loss
        return total_loss