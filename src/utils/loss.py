import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCE(nn.Module):
  def __init__(self, init_temperature=0.1):
    super().__init__()
    self.log_tau = nn.Parameter(torch.log(torch.tensor(init_temperature)))

  def forward(self, z, y):    
    tau = torch.exp(self.log_tau)
    logits = torch.matmul(z, y.T)
    logits = logits / tau

    labels = torch.arange(y.size(0)).to(z.device)

    loss = F.cross_entropy(logits, labels)
    return loss

class ProteinGOLoss(nn.Module):
  def __init__(
    self,
    weight_tensor,
    device
  ):
    super(ProteinGOLoss, self).__init__()

    self.device = device
    self.weight_tensor = torch.from_numpy(weight_tensor).float().to(self.device)

  def forward(self, y_pred, y_true):
    sig_y_pred = torch.sigmoid(y_pred)
    crossentropy_loss = self.multilabel_categorical_crossentropy(y_pred, y_true)
    go_term_centric_loss = self.weight_f1_loss(sig_y_pred, y_true, centric='go')
    protein_centric_loss = self.weight_f1_loss(sig_y_pred, y_true, centric='protein')
    total_loss = crossentropy_loss * protein_centric_loss * go_term_centric_loss
    return total_loss

  def multilabel_categorical_crossentropy(self, y_pred, y_true):

    # Modify predicted probabilities based on true labels
    y_pred = (1 - 2 * y_true) * y_pred

    # Adjust predicted probabilities
    y_pred_neg = y_pred - y_true * 1e16
    y_pred_pos = y_pred - (1 - y_true) * 1e16

    # Concatenate zeros tensor
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)

    # Compute logsumexp along the class dimension (dim=1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    total_loss = neg_loss + pos_loss

    return torch.mean(total_loss)

  def weight_f1_loss(self, y_pred, y_true, beta=1.0, centric='protein'):
    weight_tensor = self.weight_tensor.to(self.device)

    dim = 1 if centric == 'protein' else 0

    tp = torch.sum(y_true * y_pred * weight_tensor, dim=dim).to(y_pred.device)
    fp = torch.sum((1 - y_true) * y_pred * weight_tensor, dim=dim).to(y_pred.device)
    fn = torch.sum(y_true * (1 - y_pred) * weight_tensor, dim=dim).to(y_pred.device)

    precision = tp / (tp + fp + 1e-16)
    recall = tp / (tp + fn + 1e-16)

    mean_precision = torch.mean(precision)
    mean_recall = torch.mean(recall)
    f1 = self.f1_score(mean_precision, mean_recall, beta=beta)

    return 1 - f1

  def f1_score(self, precision, recall, beta=0.5, eps=1e-16):
    f1 = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + eps)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    return f1