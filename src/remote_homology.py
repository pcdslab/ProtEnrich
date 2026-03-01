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
VALID_TASKS = ['family', 'fold', 'superfamily']

def load_dataset_from_task(task):
  if task == 'family':
    dataset = load_dataset("SaeedLab/ProtEnrich", data_dir="remote_homology_family")['train']
  elif task == 'superfamily':
    dataset = load_dataset("SaeedLab/ProtEnrich", data_dir="remote_homology_superfamily")['train']
  elif task == 'fold':
    dataset = load_dataset("SaeedLab/ProtEnrich", data_dir="remote_homology_fold")['train']
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

def compute_similarity_matrix(embeddings):
  embeddings = F.normalize(embeddings, dim=1)
  return embeddings @ embeddings.T

def retrieval(embeddings, ids, k_list=(10)):
  N = embeddings.size(0)
  sim = compute_similarity_matrix(embeddings)

  acc_at_k = {k: [] for k in k_list}
  precision_at_k = {k: [] for k in k_list}
  recall_at_k = {k: [] for k in k_list}
  mrr_list = []

  for i in tqdm(range(N)):
    query = ids[i]
    scores = sim[i].clone()
    scores[i] = -1e9 # remove self-match

    relevant = []
    for j in range(N):
      if j != i and ids[j] == query:
        relevant.append(j)
    num_relevant = len(relevant)

    ranked_idx = torch.argsort(scores, descending=True)

    for k in k_list:
      topk_idx = ranked_idx[:k]
      num_correct = sum(ids[j] == query for j in topk_idx)
      acc_at_k[k].append(int(num_correct > 0))
      precision_at_k[k].append(num_correct / k)
      recall_at_k[k].append(num_correct / num_relevant)

    rank = None
    for r, j in enumerate(ranked_idx):
      if ids[j] == query:
        rank = r + 1
        break
    if rank is not None:
      mrr_list.append(1.0 / rank)
    else:
      mrr_list.append(0.0)

  results = {}
  for k in k_list:
    results[f"Acc@{k}"] = np.mean(acc_at_k[k])
    results[f"Precision@{k}"] = np.mean(precision_at_k[k])
    results[f"Recall@{k}"] = np.mean(recall_at_k[k])

  results["MRR"] = np.mean(mrr_list)

  return results

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
  ids = list(dataset['prot'])
  mapping = {name: i for i, name in enumerate(ids)}

  extract_enrich(model_name, emb_path, emb_dim_size, len(mapping), model_enrich_path, device)

  X = np.memmap(f"{emb_path}/{model_name}-enrich.mmap", dtype=np.float32, mode="r", shape=(len(mapping), emb_dim_size))
  X = torch.from_numpy(X).float().to(device)
  labels = np.array(list(dataset['label']))
  results = retrieval(X, labels)
  
  print("\n" + "="*40)
  print(f"FINAL RESULTS FOR {task.upper()} - {model_name.upper()}")
  print("="*40)
  print(f"Acc@10: {results['Acc@10']:.4f}")
  print(f"Precision@10: {results['Precision@10']:.4f}")
  print(f"Recall@10: {results['Recall@10']:.4f}")
  print("-"*5)
  print(f"MRR: {results['MRR']:.4f}")
  print("="*40)

if __name__ == '__main__':
  main()