import torch
import torch.nn as nn
import torch.nn.functional as F
import click
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, mean_squared_error
from scipy.stats import spearmanr
import optuna
from .utils import *
from .loss import ProteinGOLoss
from .model import DownstreamModel

class Trainer:
  def __init__(self, model, optimizer, train_loader, val_loader, device, criterion, checkpoint_dir, lambda_rec=0.5, lambda_seq=0.1, lambda_energy=1.0):
    self.model = model
    self.optimizer = optimizer
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.device = device
    self.model.to(self.device)
    self.criterion = criterion
    self.checkpoint_dir = checkpoint_dir
    self.lambda_rec = lambda_rec
    self.lambda_seq = lambda_seq
    self.lambda_energy = lambda_energy

  def _train_epoch(self):
    self.model.train()

    totals = {"loss": 0.0, "infonce": 0.0, "rec_struct_dyn": 0.0, "rec_seq": 0.0, "energy": 0.0}

    for inputs in tqdm(self.train_loader):
      inputs = {k: v.to(self.device) for k, v in inputs.items()}
      self.optimizer.zero_grad()

      out = self.model(inputs["seq"], inputs["struct"], inputs["dyn"])
      loss_infonce = self.criterion(out["z_seq"], out["z_struct"]) + self.criterion(out["z_seq"], out["z_dyn"])
      loss_rec_struct_dyn = F.mse_loss(out["rec_struct"], inputs["struct"]) + F.mse_loss(out["rec_dyn"], inputs["dyn"])
      loss_rec_seq = F.mse_loss(out["rec_seq"], inputs["seq"])
      loss_energy = torch.mean(out["h_algn"] ** 2)
      loss = loss_infonce + self.lambda_rec * loss_rec_struct_dyn + self.lambda_seq * loss_rec_seq + self.lambda_energy * loss_energy

      loss.backward()
      self.optimizer.step()

      totals["loss"] += loss.item()
      totals["infonce"] += loss_infonce.item()
      totals["rec_struct_dyn"] += loss_rec_struct_dyn.item()
      totals["rec_seq"] += loss_rec_seq.item()
      totals["energy"] += loss_energy.item()

    n = len(self.train_loader)

    return {k: v / n for k, v in totals.items()}

  def _eval_epoch(self):
    self.model.eval()

    totals = {"loss": 0.0, "infonce": 0.0, "rec_struct_dyn": 0.0, "rec_seq": 0.0, "energy": 0.0}

    with torch.no_grad():
      for inputs in tqdm(self.val_loader):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.model(inputs["seq"], inputs["struct"], inputs["dyn"])
        loss_infonce = self.criterion(out["z_seq"], out["z_struct"]) + self.criterion(out["z_seq"], out["z_dyn"])
        loss_rec_struct_dyn = F.mse_loss(out["rec_struct"], inputs["struct"]) + F.mse_loss(out["rec_dyn"], inputs["dyn"])
        loss_rec_seq = F.mse_loss(out["rec_seq"], inputs["seq"])
        loss_energy = torch.mean(out["h_algn"] ** 2)
        loss = loss_infonce + self.lambda_rec * loss_rec_struct_dyn + self.lambda_seq * loss_rec_seq + self.lambda_energy * loss_energy

        totals["loss"] += loss.item()
        totals["infonce"] += loss_infonce.item()
        totals["rec_struct_dyn"] += loss_rec_struct_dyn.item()
        totals["rec_seq"] += loss_rec_seq.item()
        totals["energy"] += loss_energy.item()

    n = len(self.val_loader)
    return {k: v / n for k, v in totals.items()}

  def train(self, num_epochs: int):
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
      click.echo(f"\nEpoch {epoch + 1}/{num_epochs}")

      train_metrics = self._train_epoch()
      val_metrics = self._eval_epoch()

      click.echo(
        f"Train | "
        f"Total: {train_metrics['loss']:.4f} | "
        f"InfoNCE: {train_metrics['infonce']:.4f} | "
        f"Rec(S/D): {train_metrics['rec_struct_dyn']:.4f} | "
        f"Rec(Seq): {train_metrics['rec_seq']:.4f} | "
        f"Energy: {train_metrics['energy']:.4f}"
      )

      click.echo(
        f"Val   | "
        f"Total: {val_metrics['loss']:.4f} | "
        f"InfoNCE: {val_metrics['infonce']:.4f} | "
        f"Rec(S/D): {val_metrics['rec_struct_dyn']:.4f} | "
        f"Rec(Seq): {val_metrics['rec_seq']:.4f} | "
        f"Energy: {val_metrics['energy']:.4f}"
      )

      tau = torch.exp(self.criterion.log_tau).item()
      click.echo(f"Tau: {tau:.4f}")

      if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"]
        torch.save(self.model.state_dict(), self.checkpoint_dir)
        click.echo("Model saved")

def objective(trial, config):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
  batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
  dropout = trial.suggest_float('dropout', 0.1, 0.5)

  device = config['device']
  task_type = config['task_type']
  input_dim = config['input_dim']

  train_loader = DataLoader(
    config['train_dataset'],
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=(task_type == "regression")
  )

  val_loader = DataLoader(
    config['val_dataset'],
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
  )

  model = DownstreamModel(input_dim, num_classes=config['num_classes'], dropout_rate=dropout).to(device)
  if task_type == "multilabel":
    criterion = ProteinGOLoss(weight_tensor=config['ic_loss'], device=config['device'])

  elif task_type == "regression":
    criterion = nn.MSELoss()

  elif task_type == "multiclass":
    criterion = nn.CrossEntropyLoss(weight=config['cw'].to(device))

  elif task_type == "binary" or task_type == 'pair':
    criterion = nn.CrossEntropyLoss(weight=config['cw'].to(device))

  else:
    raise ValueError("Unknown task_type")

  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

  best_metric = -np.inf if task_type == "regression" else 0.0

  for epoch in range(20):
    model.train()
    for X, y in train_loader:
      X, y = X.to(device), y.to(device)
      optimizer.zero_grad()
      outputs = model(X)
      if task_type == "regression":
        loss = criterion(outputs.squeeze(), y)
      else:
        loss = criterion(outputs, y)
      loss.backward()
      optimizer.step()

    model.eval()
    all_preds, all_gts = [], []

    with torch.no_grad():
      for X, y in val_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        if task_type == "multilabel":
          preds = torch.sigmoid(outputs)
        
        elif task_type == "regression":
          preds = outputs.squeeze()
        
        elif task_type == "multiclass":
          _, preds = torch.max(outputs, 1)

        elif task_type == "binary" or task_type == 'pair':
          preds = torch.softmax(outputs, dim=1)[:, 1]

        all_preds.append(preds.cpu().numpy())
        all_gts.append(y.cpu().numpy())

    preds_np = np.concatenate(all_preds)
    gts_np = np.concatenate(all_gts)

    if task_type == "multilabel":
      metric = evaluate_wfmax(
        preds_np,
        gts_np,
        config['ontologies_names'],
        config['ontology'],
        config['ic'],
        config['prop_map']
      )

    elif task_type == "regression":
      try:
        metric = spearmanr(gts_np, preds_np)[0]
      except:
        metric = 0.0

    elif task_type == "multiclass":
      metric = balanced_accuracy_score(gts_np, preds_np)

    elif task_type == "binary" or task_type == 'pair':
      try:
        metric = roc_auc_score(gts_np, preds_np)
      except:
        metric = 0.5

    if metric > best_metric:
      best_metric = metric

    trial.report(metric, epoch)

    if trial.should_prune():
      raise optuna.exceptions.TrialPruned()

  return best_metric


def run_multi_seed_evaluation(config, best_params, models_path):

  batch_size = best_params['batch_size']
  device = config['device']
  task_type = config['task_type']
  input_dim = config['input_dim']
  lr = best_params['lr']
  dropout = best_params['dropout']
  seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  test_results = []

  for run_idx, seed in enumerate(seeds):
    set_seed(seed)

    save_path = f'{models_path}/seed-{seed}.pt'

    train_loader = DataLoader(
      config['train_dataset'],
      batch_size=batch_size,
      shuffle=True,
      num_workers=0,
      drop_last=(task_type == "regression"))

    val_loader = DataLoader(
      config['val_dataset'],
      batch_size=batch_size,
      shuffle=False,
      num_workers=0
    )

    test_loader = DataLoader(
      config['test_dataset'],
      batch_size=batch_size,
      shuffle=False,
      num_workers=0
    )

    model = DownstreamModel(input_dim, num_classes=config['num_classes'], dropout_rate=dropout).to(device)

    if task_type == "multilabel":
      criterion = ProteinGOLoss(weight_tensor=config['ic_loss'], device=config['device'])

    elif task_type == "regression":
      criterion = nn.MSELoss()

    elif task_type == "multiclass":
      criterion = nn.CrossEntropyLoss(weight=config['cw'].to(device))

    elif task_type == "binary" or task_type == 'pair':
      criterion = nn.CrossEntropyLoss(weight=config['cw'].to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_metric = -np.inf if task_type == "regression" else 0.0

    for epoch in range(20):
      model.train()
      for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        if task_type == "regression":
          loss = criterion(outputs.squeeze(), y)
        else:
          loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

      model.eval()
      all_preds, all_gts = [], []

      with torch.no_grad():
        for X, y in val_loader:
          X, y = X.to(device), y.to(device)
          outputs = model(X)
          if task_type == "multilabel":
            preds = torch.sigmoid(outputs)
          elif task_type == "regression":
            preds = outputs.squeeze()
          elif task_type == "multiclass":
            _, preds = torch.max(outputs, 1)
          elif task_type == "binary" or task_type == 'pair':
            preds = torch.softmax(outputs, dim=1)[:, 1]

          all_preds.append(preds.cpu().numpy())
          all_gts.append(y.cpu().numpy())

      preds_np = np.concatenate(all_preds)
      gts_np = np.concatenate(all_gts)

      if task_type == "multilabel":
        metric = evaluate_wfmax(
          preds_np,
          gts_np,
          config['ontologies_names'],
          config['ontology'],
          config['ic'],
          config['prop_map']
        )

      elif task_type == "regression":
        try:
          metric = spearmanr(gts_np, preds_np)[0]
        except:
          metric = 0.0

      elif task_type == "multiclass":
        metric = balanced_accuracy_score(gts_np, preds_np)

      elif task_type == "binary" or task_type == 'pair':
        try:
          metric = roc_auc_score(gts_np, preds_np)
        except:
          metric = 0.5

      if metric > best_metric:
        best_metric = metric
        torch.save(model.state_dict(), save_path)

    best_model_run = DownstreamModel(input_dim, num_classes=config['num_classes'], dropout_rate=dropout).to(device)

    best_model_run.load_state_dict(torch.load(save_path))
    best_model_run.eval()

    test_preds = []
    test_trues = []
    with torch.no_grad():
      for X, y in test_loader:
        X = X.to(device)
        outputs = best_model_run(X)
        if task_type == 'multilabel':
          pred = torch.sigmoid(outputs)
        elif task_type == 'regression':
          pred = outputs.squeeze()
        elif task_type == 'multiclass':
          _, pred = torch.max(outputs, 1)
        elif task_type == 'binary' or task_type == 'pair':
          pred = torch.softmax(outputs, dim=1)[:, 1]
        test_preds.append(pred.cpu().numpy())
        test_trues.append(y.cpu().numpy())
    
    preds_np = np.concatenate(test_preds)
    gts_np = np.concatenate(test_trues)

    if task_type == "multilabel":
      metric = evaluate_wfmax(
        preds_np,
        gts_np,
        config['ontologies_names'],
        config['ontology'],
        config['ic'],
        config['prop_map']
      )

    elif task_type == "regression":
      metric = np.sqrt(mean_squared_error(gts_np, preds_np))

    elif task_type == "multiclass":
      metric = balanced_accuracy_score(gts_np, preds_np)

    elif task_type == "binary" or task_type == 'pair':
      try:
        metric = roc_auc_score(gts_np, preds_np)
      except:
        metric = 0.5
    
    test_results.append(metric)
  
  test_results = np.array(test_results)
  return np.mean(test_results), np.std(test_results)