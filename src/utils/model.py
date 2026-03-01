import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPEncoder(nn.Module):
  def __init__(self, in_dim, out_dim, hidden_dim=1024, n_layers=2, dropout=0.1):
    super().__init__()
    layers = []
    d = in_dim
    for _ in range(n_layers - 1):
      layers += [
        nn.Linear(d, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
      ]
      d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x)

class MultiModalProteinModel(nn.Module):
  def __init__(self, seq_dim=None, struct_dim=1024, dyn_dim=20, embed_dim=1024, project_dim=256, alpha_max=0.3):
    super().__init__()
    
    self.seq_anchor = MLPEncoder(seq_dim, embed_dim)
    self.seq_algn = MLPEncoder(seq_dim, embed_dim)
    self.struct_encoder = MLPEncoder(struct_dim, embed_dim)
    self.dyn_encoder    = MLPEncoder(dyn_dim, embed_dim)

    for p in self.struct_encoder.parameters():
      p.requires_grad = False
    for p in self.dyn_encoder.parameters():
      p.requires_grad = False

    self.seq_projector = nn.Linear(embed_dim, project_dim)
    self.struct_projector = nn.Linear(embed_dim, project_dim)
    self.dyn_projector = nn.Linear(embed_dim, project_dim)

    self.seq_decoder = MLPEncoder(embed_dim, seq_dim)
    self.struct_decoder = MLPEncoder(embed_dim, struct_dim)
    self.dyn_decoder = MLPEncoder(embed_dim, dyn_dim)

    self.alpha_logit = nn.Parameter(torch.tensor(-2.0))
    self.alpha_max = alpha_max

    self.norm_anchor = nn.LayerNorm(embed_dim)
    self.norm_algn = nn.LayerNorm(embed_dim)


  def forward(self, seq, struct=None, dyn=None):
    h_anchor = self.norm_anchor(self.seq_anchor(seq))
    h_algn = self.norm_algn(self.seq_algn(seq))

    if struct is None and dyn is None:
      rec_seq = self.seq_decoder(h_anchor)
      rec_struct = self.struct_decoder(h_algn)
      rec_dyn = self.dyn_decoder(h_algn)
      alpha = torch.sigmoid(self.alpha_logit) * self.alpha_max
      h_enrich = h_anchor + alpha * h_algn

      return {
        "rec_seq": rec_seq, "rec_struct": rec_struct, "rec_dyn": rec_dyn,
        "h_algn": h_algn, "h_anchor": h_anchor, "h_enrich": h_enrich,
      }

    with torch.no_grad():
      h_struct = self.struct_encoder(struct)
      h_dyn = self.dyn_encoder(dyn)

    z_seq = F.normalize(self.seq_projector(h_algn), dim=-1)
    z_struct = F.normalize(self.struct_projector(h_struct), dim=-1)
    z_dyn = F.normalize(self.dyn_projector(h_dyn), dim=-1)

    rec_seq = self.seq_decoder(h_anchor)
    rec_struct = self.struct_decoder(h_algn)
    rec_dyn = self.dyn_decoder(h_algn)

    alpha = torch.sigmoid(self.alpha_logit) * self.alpha_max
    h_enrich = h_anchor + alpha * h_algn

    return {
      "z_seq": z_seq, "z_struct": z_struct, "z_dyn": z_dyn, # InfoNCE
      "rec_seq": rec_seq, "rec_struct": rec_struct, "rec_dyn": rec_dyn, # MSE
      "h_algn": h_algn, # Energy
      "h_enrich": h_enrich, #  Downstream Task
    }

class DownstreamModel(nn.Module):
  def __init__(self, input_dim, num_classes=2, hidden_dim=1024, n_layers=2, dropout_rate=0.1):
    super(DownstreamModel, self).__init__()
    layers = []

    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.BatchNorm1d(hidden_dim))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout_rate))
    
    for _ in range(n_layers - 1):
      layers.append(nn.Linear(hidden_dim, hidden_dim))
      layers.append(nn.BatchNorm1d(hidden_dim))
      layers.append(nn.ReLU())
      layers.append(nn.Dropout(dropout_rate))

    layers.append(nn.Linear(hidden_dim, num_classes))
    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x)