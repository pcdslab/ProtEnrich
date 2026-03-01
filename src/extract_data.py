import torch
from torch.nn.functional import softmax
from torch.cuda.amp import autocast
from rocketshp import RocketSHP, load_sequence, load_structure
from rocketshp.structure.protein_chain import ProteinChain
from rocketshp.features import esm3_sequence, esm3_vqvae
from rocketshp.esm3 import get_model, get_tokenizers, get_structure_vae
from biotite.structure.io import pdb
from biotite.structure import to_sequence
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from transformers import T5Tokenizer, T5EncoderModel, AutoModel, AutoTokenizer, AutoModelForCausalLM
import re
import numpy as np
from tqdm import tqdm
import os
from datasets import load_dataset
import click
from Bio import SeqIO
import sys
from sequence_models.pretrained import load_model_and_alphabet
from utils.extract import *

VALID_MODELS = ['structure', 'dynamics', 'ankh3', 'carp_640m', 'esm1b', 'esm2_t36', 'esmc_600m', 'progen2', 'protbert', 'prott5']
DATASETS = ['pretraining', 'out-of-distribution']

@click.command()
@click.option('--task', 
  required=True,
  type=click.Choice(VALID_MODELS),
  help="Task - structure, dynamics, or sequence models."
)
@click.option('--dataset_name', 
  required=True,
  type=click.Choice(DATASETS),
  help="Dataset name - pretraining or out-of-distribution.",
  default="pretraining"
)

def main(task, dataset_name):
  if dataset_name == 'pretraining':
    dataset = load_dataset("SaeedLab/ProtEnrich", data_dir="pretraining")['train'][:10]
    out_path = '../embs_pretraining'
    os.makedirs('../embs_pretraining', exist_ok=True)
  elif dataset_name == 'out-of-distribution':
    dataset = load_dataset("SaeedLab/ProtEnrich", data_dir="out-of-distribution")['train'][:10]
    out_path = '../embs_out_of_distribution'
    os.makedirs('../embs_out_of_distribution', exist_ok=True)
  else:
    raise ValueError(f"Invalid dataset: {dataset_name}")
  
  if task == 'structure':
    extract_structure(dataset, out_path=out_path)
  elif task == 'dynamics':
    extract_dyn(dataset, out_path=out_path)
  elif task == 'ankh3':
    extract_ankh_3_xl(dataset, out_path=out_path)
  elif task == 'carp_640m':
    extract_carp_640m(dataset, out_path=out_path)
  elif task == 'esm1b':
    extract_esm1b(dataset, out_path=out_path)
  elif task == 'esm2_t36':
    extract_esm2_t36(dataset, out_path=out_path)
  elif task == 'esmc_600m':
    extract_esmc_600m(dataset, out_path=out_path)
  elif task == 'progen2':
    extract_progen2(dataset, out_path=out_path)
  elif task == 'protbert':
    extract_protbert(dataset, out_path=out_path)
  elif task == 'prott5':
    extract_prott5(dataset, out_path=out_path)
  else:
    raise ValueError(f"Invalid task: {task}")

if __name__ == '__main__':
  main()