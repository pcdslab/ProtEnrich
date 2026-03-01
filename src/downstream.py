import click
import os
import torch
import numpy as np
import pandas as pd
import optuna
from datasets import load_dataset, concatenate_datasets, Dataset
import ast
from sklearn.utils.class_weight import compute_class_weight
from huggingface_hub import hf_hub_download

from utils.utils import *
from utils.trainer import objective, run_multi_seed_evaluation
from utils.model import DownstreamModel, MultiModalProteinModel
from utils.loss import ProteinGOLoss
from utils.dataset import DownstreamDataset
from utils.extract import *

task_type = {
  'fluorescence': 'regression',
  'localization_bin': 'binary',
  'localization_multi': 'multiclass',
  'metal_ion_binding': 'binary',
  'ppi': 'pair',
  'protein_function_bp': 'multilabel',
  'protein_function_cc': 'multilabel',
  'protein_function_mf': 'multilabel',
  'solubility': 'binary',
  'thermostability': 'binary'
}
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
VALID_TASKS = ['fluorescence', 'localization_bin', 'localization_multi', 'metal_ion_binding', 'ppi', 'protein_function_bp', 'protein_function_cc', 'protein_function_mf', 'solubility', 'thermostability']

def load_dataset_from_task(task):
  if task == 'fluorescence':
    dataset = load_dataset("SaeedLab/ProtEnrich", data_dir="fluorescence")
  elif task == 'localization_bin':
    dataset = load_dataset("SaeedLab/ProtEnrich", data_dir="localization_bin")
  elif task == 'localization_multi':
    dataset = load_dataset("SaeedLab/ProtEnrich", data_dir="localization_multi")
  elif task == 'metal_ion_binding':
    dataset = load_dataset("SaeedLab/ProtEnrich", data_dir="metal_ion_binding")
  elif task == 'ppi':
    dataset = load_dataset("SaeedLab/ProtEnrich", data_dir="ppi")
  elif task == 'protein_function_bp':
    dataset = load_dataset("SaeedLab/ProtEnrich", data_dir="protein_function_bp")
  elif task == 'protein_function_cc':
    dataset = load_dataset("SaeedLab/ProtEnrich", data_dir="protein_function_cc")
  elif task == 'protein_function_mf':
    dataset = load_dataset("SaeedLab/ProtEnrich", data_dir="protein_function_mf")
  elif task == 'solubility':
    dataset = load_dataset("SaeedLab/ProtEnrich", data_dir="solubility")
  elif task == 'thermostability':
    dataset = load_dataset("SaeedLab/ProtEnrich", data_dir="thermostability")
  else:
    raise ValueError(f"Invalid task: {task}")
  return Dataset.from_dict(dataset['train'][:10]), Dataset.from_dict(dataset['validation'][:10]), Dataset.from_dict(dataset['test'][:10])

def get_all_sequences(train, val, test):
  full_dataset = concatenate_datasets([train, val, test])
  cols = full_dataset.column_names
  if "sequence" in cols:
    seqs = list(full_dataset["sequence"])
    ids = list(full_dataset['prot'])
  elif "sequence_a" in cols:
    seqs = list(full_dataset["sequence_a"]) + list(full_dataset["sequence_b"])
    seqs = list(set(seqs))
    ids = list(full_dataset["prot_a"]) + list(full_dataset["prot_b"])
    ids = sorted(list(set(ids)))
  else:
    raise ValueError("Unknown dataset format")
  mapping = {name: i for i, name in enumerate(ids)}
  return Dataset.from_dict({'sequence': seqs}), mapping

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

def parse_terms(example):
  example["terms"] = ast.literal_eval(example["terms"])
  return example

@click.command()
@click.option('--task', required=True, type=click.Choice(VALID_TASKS))
@click.option('--model_name', required=True, type=click.Choice(VALID_MODELS))
def main(task, model_name):
  emb_path = f'../embs_{task}_{model_name}'
  models_path = f'../models_{task}_{model_name}'
  model_enrich_path = f'../models/{model_name}-enrich.pt'

  os.makedirs(emb_path, exist_ok=True)
  os.makedirs(models_path, exist_ok=True)

  task_type_running = task_type[task]
  emb_dim_size = emb_dim[model_name]

  set_seed(42)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  train, validation, test = load_dataset_from_task(task)
  all_sequences, mapping = get_all_sequences(train, validation, test)
  generate_embs(all_sequences, model_name, emb_path)
  extract_enrich(model_name, emb_path, emb_dim_size, len(mapping), model_enrich_path, device)

  if task_type_running == 'pair':
    emb_dim_size *= 2

  config = {
    "device": device,
    "task_type": task_type_running,
    "input_dim": emb_dim_size
  }

  if 'protein_function' in task:
    train = train.map(parse_terms)
    validation = validation.map(parse_terms)
    test = test.map(parse_terms)

    ic_ont = load_dataset("SaeedLab/ProtEnrich", data_files={"ic": f"{task}/ic.parquet"})['ic'].to_pandas()
    ic = ic_ont.set_index('terms')['IC'].to_dict()
    ontologies_names = ic_ont['terms'].values.tolist()
    mapper = {k: i for i, k in enumerate(ontologies_names)}

    obo_path = hf_hub_download(repo_id="SaeedLab/ProtEnrich", filename=f"{task}/go-basic.obo", repo_type="dataset")
    ontology = generate_ontology(obo_path)
    prop_map = get_propagation_map(ontologies_names, ontology)

    train_dataset = DownstreamDataset(names=list(train['prot']),
                                      labels=list(train['terms']),
                                      task_type=task_type_running,
                                      emb_dim=emb_dim_size,
                                      mapping=mapping,
                                      emb_dir=emb_path,
                                      mode=model_name + '-enrich',
                                      mapper=mapper)

    val_dataset = DownstreamDataset(names=list(validation['prot']),
                                    labels=list(validation['terms']),
                                    task_type=task_type_running,
                                    emb_dim=emb_dim_size,
                                    mapping=mapping,
                                    emb_dir=emb_path,
                                    mode=model_name + '-enrich',
                                    mapper=mapper)

    test_dataset = DownstreamDataset(names=list(test['prot']),
                                     labels=list(test['terms']),
                                     task_type=task_type_running,
                                     emb_dim=emb_dim_size,
                                     mapping=mapping,
                                     emb_dir=emb_path,
                                     mode=model_name + '-enrich',
                                     mapper=mapper)

    config.update({
      "train_dataset": train_dataset,
      "val_dataset": val_dataset,
      "test_dataset": test_dataset,
      "num_classes": len(ontologies_names),
      "ontologies_names": ontologies_names,
      "ontology": ontology,
      "ic": ic,
      "ic_loss": ic_ont.IC.values,
      "prop_map": prop_map
    })

  elif task_type_running == "regression":
    train_dataset = DownstreamDataset(names=list(train['prot']),
                                      labels=list(train['label']),
                                      task_type=task_type_running,
                                      emb_dim=emb_dim_size,
                                      mapping=mapping,
                                      emb_dir=emb_path,
                                      mode=model_name + '-enrich')

    val_dataset = DownstreamDataset(names=list(validation['prot']),
                                    labels=list(validation['label']),
                                    task_type=task_type_running,
                                    emb_dim=emb_dim_size,
                                    mapping=mapping,
                                    emb_dir=emb_path,
                                    mode=model_name + '-enrich')

    test_dataset = DownstreamDataset(names=list(test['prot']),
                                     labels=list(test['label']),
                                     task_type=task_type_running,
                                     emb_dim=emb_dim_size,
                                     mapping=mapping,
                                     emb_dir=emb_path,
                                     mode=model_name + '-enrich')

    config.update({
      "train_dataset": train_dataset,
      "val_dataset": val_dataset,
      "test_dataset": test_dataset,
      "num_classes": 1
    })

  elif task_type_running in ["multiclass", "binary"]:

    cw_values = compute_class_weight('balanced', classes=np.unique(list(train['label'])), y=list(train['label']))
    cw = torch.tensor(cw_values, dtype=torch.float).to(device)

    train_dataset = DownstreamDataset(names=list(train['prot']),
                                      labels=list(train['label']),
                                      task_type=task_type_running,
                                      emb_dim=emb_dim_size,
                                      mapping=mapping,
                                      emb_dir=emb_path,
                                      mode=model_name + '-enrich')

    val_dataset = DownstreamDataset(names=list(validation['prot']),
                                    labels=list(validation['label']),
                                    task_type=task_type_running,
                                    emb_dim=emb_dim_size,
                                    mapping=mapping,
                                    emb_dir=emb_path,
                                    mode=model_name + '-enrich')

    test_dataset = DownstreamDataset(names=list(test['prot']),
                                     labels=list(test['label']),
                                     task_type=task_type_running,
                                     emb_dim=emb_dim_size,
                                     mapping=mapping,
                                     emb_dir=emb_path,
                                     mode=model_name + '-enrich')

    config.update({
      "train_dataset": train_dataset,
      "val_dataset": val_dataset,
      "test_dataset": test_dataset,
      "cw": cw,
      "num_classes": len(np.unique(list(train['label'])))
    })

  elif task_type_running == "pair":

    cw_values = compute_class_weight('balanced', classes=np.unique(list(train['label'])), y=list(train['label']))
    cw = torch.tensor(cw_values, dtype=torch.float).to(device)

    train_dataset = DownstreamDataset(names=list(train['prot_a']),
                                      names_b=list(train['prot_b']),
                                      labels=list(train['label']),
                                      task_type=task_type_running,
                                      emb_dim=emb_dim_size,
                                      mapping=mapping,
                                      emb_dir=emb_path,
                                      mode=model_name + '-enrich')

    val_dataset = DownstreamDataset(names=list(validation['prot_a']),
                                    names_b=list(validation['prot_b']),
                                    labels=list(validation['label']),
                                    task_type=task_type_running,
                                    emb_dim=emb_dim_size,
                                    mapping=mapping,
                                    emb_dir=emb_path,
                                    mode=model_name + '-enrich')

    test_dataset = DownstreamDataset(names=list(test['prot_a']),
                                     names_b=list(test['prot_b']),
                                     labels=list(test['label']),
                                     task_type=task_type_running,
                                     emb_dim=emb_dim_size,
                                     mapping=mapping,
                                     emb_dir=emb_path,
                                     mode=model_name + '-enrich')

    config.update({
      "train_dataset": train_dataset,
      "val_dataset": val_dataset,
      "test_dataset": test_dataset,
      "cw": cw,
      "num_classes": len(np.unique(list(train['label'])))
    })


  study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
  study.optimize(lambda trial: objective(trial, config), n_trials=1)
  best_params = study.best_trial.params

  mean_results, std_results = run_multi_seed_evaluation(config=config, best_params=best_params, models_path=models_path)

  print("\n" + "="*40)
  print(f"FINAL RESULTS FOR {task.upper()} - {model_name.upper()}")
  print("="*40)
  if task_type_running == 'regression':
    print(f"RMSE: {mean_results:.4f} ± {std_results:.4f}")
  elif task_type_running == 'binary' or task_type_running == 'pair':
    print(f"AUCROC: {mean_results:.4f} ± {std_results:.4f}")
  elif task_type_running == 'multiclass':
    print(f"Balanced Accuracy: {mean_results:.4f} ± {std_results:.4f}")
  elif task_type_running == 'multilabel':
    print(f"wFmax: {mean_results:.4f} ± {std_results:.4f}")
  print("="*40)

if __name__ == '__main__':
  main()