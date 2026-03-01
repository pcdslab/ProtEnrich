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
from .model import MultiModalProteinModel

# ============
# STRUCTURE
# ============

def extract_structure(dataset, out_path):
  EMB_DIM = 1024
  MAX_LEN = 1022

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5_fp16', do_lower_case=False)
  model = T5EncoderModel.from_pretrained("Rostlab/ProstT5_fp16").to(device).eval()
  
  X = np.memmap(f'{out_path}/struct.mmap', dtype=np.float32, mode='w+', shape=(len(dataset['structure']), EMB_DIM))

  with torch.inference_mode():
    for i, seq in tqdm(enumerate(dataset['structure']), total=len(dataset['structure'])):
      sequence_examples = [seq[:MAX_LEN]]
      sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
      sequence_examples = ["<fold2AA>" + " " + s for s in sequence_examples]
      ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, return_tensors='pt').to(device)
      embedding_rpr = model(ids.input_ids, attention_mask=ids.attention_mask)
      embs = embedding_rpr.last_hidden_state[0, 1:-1].detach().cpu().numpy().mean(axis=0)
      X[i] = embs

  X.flush()

# ============
# DYNAMICS
# ============

def extract_dyn(dataset, out_path):
  EMB_DIM = 20
  MAX_LEN = 1022
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  def truncate_protein_chain(chain, max_len=1022):
    if len(chain.sequence) <= max_len:
      return chain
    return ProteinChain(
      id=chain.id,
      sequence=chain.sequence[:max_len],
      chain_id=chain.chain_id,
      entity_id=chain.entity_id,
      residue_index=chain.residue_index[:max_len],
      insertion_code=chain.insertion_code[:max_len],
      atom37_positions=chain.atom37_positions[:max_len],
      atom37_mask=chain.atom37_mask[:max_len],
      confidence=chain.confidence[:max_len],
    )

  model = RocketSHP.load_from_checkpoint("latest").to(device).eval()
  esm_model, esm_tokenizers = (get_model(device=device), get_tokenizers())
  esm_model = esm_model.to(device)
  esm_structure_model = get_structure_vae(device=device)
  esm_structure_model = esm_structure_model.to(device)

  X = np.memmap(f'{out_path}/dyn.mmap', dtype=np.float32, mode='w+', shape=(len(dataset['file']), EMB_DIM))

  for i, file_name in tqdm(enumerate(dataset['file']), total=len(dataset['file'])):
    structure_file = f'../pdb_files/{file_name}.pdb'
    structure = pdb.PDBFile.read(structure_file).get_structure()
    chain = ProteinChain.from_atomarray(structure)
    chain = truncate_protein_chain(chain, MAX_LEN)
    sequence = chain.sequence

    with torch.no_grad(), autocast(dtype=torch.float16):
      seq_features = esm3_sequence(sequence=sequence, model=esm_model, tokenizers=esm_tokenizers)[:, 1:-1, :]
      struct_features = esm3_vqvae(chain=chain, esm_struct_encoder=esm_structure_model, stage="encoded")
      dynamics_pred = model({
        "seq_feats": seq_features,
        "struct_feats": struct_features
      })

    embs = dynamics_pred['shp'][0].mean(dim=0).float().cpu().numpy()
    X[i] = embs

  X.flush()

# ============
# ANKH 3 XL
# ============

def extract_ankh_3_xl(dataset, out_path):
  EMB_DIM = 2560
  MAX_LEN = 1022

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  tokenizer = T5Tokenizer.from_pretrained("ElnaggarLab/ankh3-xl")
  model = T5EncoderModel.from_pretrained("ElnaggarLab/ankh3-xl").to(device).eval()

  X = np.memmap(f'{out_path}/ankh3.mmap', dtype=np.float32, mode='w+', shape=(len(dataset['sequence']), EMB_DIM))

  with torch.inference_mode():
    for i, seq in tqdm(enumerate(dataset['sequence']), total=len(dataset['sequence'])):
      protein = "[NLU]" + seq[:MAX_LEN]
      ids = tokenizer.batch_encode_plus([protein], add_special_tokens=True, return_tensors='pt', is_split_into_words=False).to(device)
      embedding_rpr = model(**ids)
      embs = embedding_rpr.last_hidden_state[0, 1:-1].detach().cpu().numpy().mean(axis=0)
      X[i] = embs
  
  X.flush()

# ============
# CARP 640M
# ============

def extract_carp_640m(dataset, out_path):
  EMB_DIM = 1280
  MAX_LEN = 1022

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  model, collater = load_model_and_alphabet('carp_640M')
  model.to(device)

  X = np.memmap(f'{out_path}/carp_640m.mmap', dtype=np.float32, mode='w+', shape=(len(dataset['sequence']), EMB_DIM))

  with torch.inference_mode():
    for i, seq in tqdm(enumerate(dataset['sequence']), total=len(dataset['sequence'])):
      protein = "".join(seq[:MAX_LEN])
      x = collater([[protein]])[0]
      x = x.to(device)
      rep = model(x)
      embs = rep['representations'][56][0].mean(axis=0)
      X[i] = embs.cpu().numpy()
  
  X.flush()

# ============
# ESM1b
# ============

def extract_esm1b(dataset, out_path):
  EMB_DIM = 1280
  MAX_LEN = 1022

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  tokenizer = AutoTokenizer.from_pretrained('facebook/esm1b_t33_650M_UR50S')
  model = AutoModel.from_pretrained("facebook/esm1b_t33_650M_UR50S").to(device).eval()

  X = np.memmap(f'{out_path}/esm1b.mmap', dtype=np.float32, mode='w+', shape=(len(dataset['sequence']), EMB_DIM))

  with torch.inference_mode():
    for i, seq in tqdm(enumerate(dataset['sequence']), total=len(dataset['sequence'])):
      protein = " ".join(seq[:MAX_LEN])
      ids = tokenizer.batch_encode_plus([protein], add_special_tokens=True, return_tensors='pt').to(device)
      embedding_rpr = model(**ids)
      embs = embedding_rpr.last_hidden_state[0, 1:-1].detach().cpu().numpy().mean(axis=0)
      X[i] = embs
  
  X.flush()

# ============
# ESM2 T36
# ============

def extract_esm2_t36(dataset, out_path):
  EMB_DIM = 2560
  MAX_LEN = 1022

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t36_3B_UR50D')
  model = AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D").to(device).eval()

  X = np.memmap(f'{out_path}/esm2_t36.mmap', dtype=np.float32, mode='w+', shape=(len(dataset['sequence']), EMB_DIM))

  with torch.inference_mode():
    for i, seq in tqdm(enumerate(dataset['sequence']), total=len(dataset['sequence'])):
      protein = " ".join(seq[:MAX_LEN])
      ids = tokenizer.batch_encode_plus([protein], add_special_tokens=True, return_tensors='pt').to(device)
      embedding_rpr = model(**ids)
      embs = embedding_rpr.last_hidden_state[0, 1:-1].detach().cpu().numpy().mean(axis=0)
      X[i] = embs
  
  X.flush()

# ============
# ESM C 600M
# ============

def extract_esmc_600m(dataset, out_path):
  EMB_DIM = 1152
  MAX_LEN = 1022

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  client = ESMC.from_pretrained("esmc_600m").to(device).eval()

  X = np.memmap(f'{out_path}/esmc_600m.mmap', dtype=np.float32, mode='w+', shape=(len(dataset['sequence']), EMB_DIM))

  with torch.inference_mode():
    for i, seq in tqdm(enumerate(dataset['sequence']), total=len(dataset['sequence'])):
      protein = ESMProtein(sequence=seq[:MAX_LEN])
      protein_tensor = client.encode(protein)
      logits_output = client.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
      )
      embs = logits_output.embeddings[0, 1:-1].detach().cpu().numpy().mean(axis=0)
      X[i] = embs
  
  X.flush()

# ============
# ProGen2
# ============

def extract_progen2(dataset, out_path):
  EMB_DIM = 1536
  MAX_LEN = 1022

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  tokenizer = AutoTokenizer.from_pretrained('hugohrban/progen2-base', trust_remote_code=True)
  model = AutoModelForCausalLM.from_pretrained("hugohrban/progen2-base",  trust_remote_code=True).to(device).eval()

  X = np.memmap(f'{out_path}/progen2.mmap', dtype=np.float32, mode='w+', shape=(len(dataset['sequence']), EMB_DIM))

  with torch.inference_mode():
    for i, seq in tqdm(enumerate(dataset['sequence']), total=len(dataset['sequence'])):
      protein = "".join(seq[:MAX_LEN])
      ids = tokenizer.batch_encode_plus([protein], return_tensors='pt').to(device)
      embedding_rpr = model(**ids, output_hidden_states=True)
      embs = embedding_rpr.hidden_states[-1][0].detach().cpu().numpy().mean(axis=0)
      X[i] = embs
  
  X.flush()

# ============
# ProtBERT
# ============

def extract_protbert(dataset, out_path):
  EMB_DIM = 1024
  MAX_LEN = 1022

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert')
  model = AutoModel.from_pretrained("Rostlab/prot_bert").to(device).eval()

  X = np.memmap(f'{out_path}/protbert.mmap', dtype=np.float32, mode='w+', shape=(len(dataset['sequence']), EMB_DIM))

  with torch.inference_mode():
    for i, seq in tqdm(enumerate(dataset['sequence']), total=len(dataset['sequence'])):
      protein = " ".join(seq[:MAX_LEN])
      ids = tokenizer.batch_encode_plus([protein], add_special_tokens=True, return_tensors='pt').to(device)
      embedding_rpr = model(**ids)
      embs = embedding_rpr.last_hidden_state[0, 1:-1].detach().cpu().numpy().mean(axis=0)
      X[i] = embs
  
  X.flush()

# ============
# ProtT5
# ============

def extract_prott5(dataset, out_path):
  EMB_DIM = 1024
  MAX_LEN = 1022

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc')
  model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc').to(device).eval()

  X = np.memmap(f'{out_path}/prott5.mmap', dtype=np.float32, mode='w+', shape=(len(dataset['sequence']), EMB_DIM))

  with torch.inference_mode():
    for i, seq in tqdm(enumerate(dataset['sequence']), total=len(dataset['sequence'])):
      protein = " ".join(seq[:MAX_LEN])
      ids = tokenizer.batch_encode_plus([protein], add_special_tokens=True, return_tensors='pt').to(device)
      embedding_rpr = model(**ids)
      embs = embedding_rpr.last_hidden_state[0, :-1].detach().cpu().numpy().mean(axis=0)
      X[i] = embs
  
  X.flush()

# ============
# Downstream
# ============

def extract_enrich(model_name, out_path, emb_dim, input_size, model_enrich_path, device):
  X = np.memmap(f'{out_path}/{model_name}.mmap', dtype=np.float32, mode='r', shape=(input_size, emb_dim))
  X_enrich = np.memmap(f'{out_path}/{model_name}-enrich.mmap', dtype=np.float32, mode="w+", shape=(input_size, 1024))

  model_enrich = MultiModalProteinModel(seq_dim=emb_dim).to(device)
  model_enrich.load_state_dict(torch.load(model_enrich_path, weights_only=True))
  model_enrich.eval()

  with torch.no_grad():
    for i in tqdm(range(input_size)):
      this_seq = torch.from_numpy(X[i]).unsqueeze(0).to(device)
      z = model_enrich(this_seq)
      h_enrich = z['h_enrich']
      X_enrich[i] = h_enrich.squeeze(0).cpu().numpy()
  
  X_enrich.flush()

# ============
# Generate
# ============

def generate_data(model_name, out_path, emb_dim, input_size, model_enrich_path, device):
  X = np.memmap(f'{out_path}/{model_name}.mmap', dtype=np.float32, mode='r', shape=(input_size, emb_dim))
  X_struct = np.memmap(f'{out_path}/{model_name}-struct.mmap', dtype=np.float32, mode="w+", shape=(input_size, 1024))
  X_dyn = np.memmap(f'{out_path}/{model_name}-dyn.mmap', dtype=np.float32, mode="w+", shape=(input_size, 20))

  model_enrich = MultiModalProteinModel(seq_dim=emb_dim).to(device)
  model_enrich.load_state_dict(torch.load(model_enrich_path, weights_only=True))
  model_enrich.eval()

  with torch.no_grad():
    for i in tqdm(range(input_size)):
      this_seq = torch.from_numpy(X[i]).unsqueeze(0).to(device)
      z = model_enrich(this_seq)
      X_struct[i] = z['rec_struct'].squeeze(0).cpu().numpy()
      X_dyn[i] = z['rec_dyn'].squeeze(0).cpu().numpy()
  
  X_struct.flush()
  X_dyn.flush()