import torch
import torch.nn.functional as F

def calculate_metrics(logits, targets):
        """Calculates loss, accuracy, and perplexity score for a batch."""
        logits_flat = logits.flatten(0, 1)
        targets_flat = targets.flatten()
        loss = F.cross_entropy(logits_flat, targets_flat)
        perplexity = torch.exp(loss)
        predicted_labels = torch.argmax(logits_flat, dim=1)
        accuracy = (predicted_labels == targets_flat).sum().item() / targets_flat.size(0)
        return loss, accuracy, perplexity