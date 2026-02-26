import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

def dice_coefficient(preds, targets, threshold=0.5, eps=1e-6):

    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    dice = (2. * intersection + eps) / (preds.sum() + targets.sum() + eps)

    return dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=None):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_score = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice_score

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        return dice_loss + BCE
    
def plot_confusion_matrix(cm, class_names):
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Lung Cancer Detection: Confusion Matrix')
    plt.show()