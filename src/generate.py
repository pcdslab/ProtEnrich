import click
import os
import torch
import numpy as np
import pandas as pd
import optuna
from datasets import load_dataset, concatenate_datasets, Dataset
import ast
from sklearn.utils.class_weight import compute_class_weight

from utils.utils import *
from utils.model import MultiModalProteinModel
from utils.extract import *

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

VALID_MODELS = ['ankh3', 'carp_640m', 'esm1b', 'esm2_t36', 'esmc_600m', 'progen2', 'protbert', 'prott5']
VALID_TASKS = ['out-of-distribution']

def load_dataset_from_task(task):
  if task == 'out-of-distribution':
    dataset = load_dataset("SaeedLab/ProtEnrich", data_dir='out-of-distribution')
  else:
    raise ValueError(f"Invalid task: {task}")
  return Dataset.from_dict(dataset['train'][:10])

def generate_embs(dataset, model_name, out_path):
  if model_name == 'ankh3':
    extract_ankh_3_xl(dataset, out_path=out_path)
  elif model_name == 'carp_640m':
    extract_carp_640m(dataset, out_path=out_path)
  elif model_name == 'esm1b':
    extract_esm1b(dataset, out_path=out_path)
  elif model_name == 'esm2_t36':
    extract_esm2_t36(dataset, out_path=out_path)
  elif model_name == 'esmc_600m':
    extract_esmc_600m(dataset, out_path=out_path)
  elif model_name == 'progen2':
    extract_progen2(dataset, out_path=out_path)
  elif model_name == 'protbert':
    extract_protbert(dataset, out_path=out_path)
  elif model_name == 'prott5':
    extract_prott5(dataset, out_path=out_path)
  else:
    raise ValueError(f"Invalid model name: {model_name}")

@click.command()
@click.option('--task', required=True, type=click.Choice(VALID_TASKS))
@click.option('--model_name', required=True, type=click.Choice(VALID_MODELS))
def main(task, model_name):
  emb_path = f'../embs_{task}_{model_name}'
  model_enrich_path = f'../models/{model_name}-enrich.pt'
  os.makedirs(emb_path, exist_ok=True)

  emb_dim_size = emb_dim[model_name]
  set_seed(42)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  dataset = load_dataset_from_task(task)
  generate_embs(dataset, model_name, emb_path)

  seqs = list(dataset["sequence"])
  ids = list(dataset['prot_id'])
  mapping = {name: i for i, name in enumerate(ids)}

  generate_data(model_name, emb_path, emb_dim_size, len(mapping), model_enrich_path, device)

  struct = np.memmap(f"{emb_path}/{model_name}-struct.mmap", dtype=np.float32, mode="r", shape=(len(mapping), 1024))
  dyn = np.memmap(f"{emb_path}/{model_name}-dyn.mmap", dtype=np.float32, mode="r", shape=(len(mapping), 20))

  print(f'Structure data: {struct.shape}')
  print(f'Dynamics data: {dyn.shape}')
  

if __name__ == '__main__':
  main()