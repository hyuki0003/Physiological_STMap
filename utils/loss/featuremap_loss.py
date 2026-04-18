import torch
import torch.nn as nn
import torch.fft


class NegPearsonLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(NegPearsonLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        # Flatten the image (64, 1, 16, 160) -> (64, 2560)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # Normalize by subtracting the mean
        pred_flat = pred_flat - pred_flat.mean(dim=1, keepdim=True)
        target_flat = target_flat - target_flat.mean(dim=1, keepdim=True)

        # Calculate norms and avoid division by zero
        pred_norm = torch.norm(pred_flat, p=2, dim=1) + self.epsilon
        target_norm = torch.norm(target_flat, p=2, dim=1) + self.epsilon

        # Calculate cosine similarity
        cosine_similarity = torch.sum(pred_flat * target_flat, dim=1) / (pred_norm * target_norm)

        # Negate the Pearson correlation (1 - correlation)
        neg_pearson_loss = 1 - cosine_similarity.mean()

        return neg_pearson_loss

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
        self.NegPearsonLoss = NegPearsonLoss()
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
