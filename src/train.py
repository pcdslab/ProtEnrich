from datasets import load_dataset
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import os
import warnings
import json
import gc
import click

from utils.model import MultiModalProteinModel
from utils.trainer import Trainer
from utils.dataset import MultiModalProteinDataset
from utils.utils import set_seed
from utils.loss import InfoNCE

set_seed(42)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.makedirs('../models', exist_ok=True)
VALID_MODELS = ['ankh3', 'carp_640m', 'esm1b', 'esm2_t36', 'esmc_600m', 'progen2', 'protbert', 'prott5']
emb_dim = {
  'ankh3': 2560,
  'carp_640m': 1280,
  'esm1b': 1280,
  'esm2_t36': 2560,
  'esmc_600m': 1152,
  'progen2': 1536,
  'protbert': 1024,
  'prott5': 1024,
}

@click.command()
@click.option('--model_name', 
  required=True,
  type=click.Choice(VALID_MODELS),
  help="Task - structure, dynamics, or sequence models."
)
def main(model_name):
  dataset = load_dataset("SaeedLab/ProtEnrich", data_dir="pretraining")['train'][:10]
  
  dataset_length = list(range(len(dataset['prot_id'])))
  random.seed(42) 
  random.shuffle(dataset_length)
  split_idx = int(len(dataset_length) * 0.95)
  train_indices = dataset_length[:split_idx]
  val_indices = dataset_length[split_idx:]
  
  train_dataset = MultiModalProteinDataset(indices=train_indices, 
                                           model_name=model_name, 
                                           emb_dim=emb_dim[model_name],
                                           total_length=len(dataset_length))
  val_dataset = MultiModalProteinDataset(indices=val_indices, 
                                         model_name=model_name, 
                                         emb_dim=emb_dim[model_name],
                                         total_length=len(dataset_length))

  train_loader = DataLoader(
    train_dataset, 
    batch_size=512, 
    shuffle=True, 
    num_workers=4,
    pin_memory=True, 
    persistent_workers=True, 
    prefetch_factor=2
  )
  
  val_loader = DataLoader(
    val_dataset, 
    batch_size=512, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
  )
  
  model = MultiModalProteinModel(seq_dim=emb_dim[model_name]).to(device)
  criterion = InfoNCE() 
  lr = 1e-4
  optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=lr)

  checkpoint_dir = f"../models/{model_name}-enrich.pt"

  trainer = Trainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    checkpoint_dir=checkpoint_dir,
    criterion=criterion
  )

  trainer.train(num_epochs=1)

if __name__ == '__main__':
  main()