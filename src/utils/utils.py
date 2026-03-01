import random
import numpy as np
import torch

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def get_ancestors(ontology, term):
  list_of_terms = []
  list_of_terms.append(term)
  data = []

  while len(list_of_terms) > 0:
    new_term = list_of_terms.pop(0)

    if new_term not in ontology:
      break
    data.append(new_term)
    for parent_term in ontology[new_term]['parents']:
      if parent_term in ontology:
        list_of_terms.append(parent_term)

  return data

def generate_ontology(file, specific_space=False, name_specific_space=''):
  ontology = {}
  gene = {}
  flag = False
  with open(file) as f:
    for line in f.readlines():
      line = line.replace('\n','')
      if line == '[Term]':
        if 'id' in gene:
          ontology[gene['id']] = gene
        gene = {}
        gene['parents'], gene['alt_ids'] = [], []
        flag = True

      elif line == '[Typedef]':
        flag = False

      else:
        if not flag:
          continue
        items = line.split(': ')
        if items[0] == 'id':
          gene['id'] = items[1]
        elif items[0] == 'alt_id':
          gene['alt_ids'].append(items[1])
        elif items[0] == 'namespace':
          if specific_space:
            if name_specific_space == items[1]:
              gene['namespace'] = items[1]
            else:
              gene = {}
              flag = False
          else:
            gene['namespace'] = items[1]
        elif items[0] == 'is_a':
          gene['parents'].append(items[1].split(' ! ')[0])
        elif items[0] == 'name':
          gene['name'] = items[1]
        elif items[0] == 'is_obsolete':
          gene = {}
          flag = False

    key_list = list(ontology.keys())
    for key in key_list:
      ontology[key]['ancestors'] = get_ancestors(ontology, key)
      for alt_ids in ontology[key]['alt_ids']:
        ontology[alt_ids] = ontology[key]

    for key, value in ontology.items():
      if 'children' not in value:
        value['children'] = []
      for p_id in value['parents']:
        if p_id in ontology:
          if 'children' not in ontology[p_id]:
            ontology[p_id]['children'] = []
          ontology[p_id]['children'].append(key)

  return ontology

def get_propagation_map(ontologies_names, ontology):
  name_to_idx = {name: i for i, name in enumerate(ontologies_names)}
  prop_map = []
    
  for term in ontologies_names:
    parents_indices = []
    if term in ontology:
      for parent in ontology[term]['ancestors']:
        if parent in name_to_idx:
          parents_indices.append(name_to_idx[parent])
    prop_map.append(parents_indices)
        
  return prop_map

def propagate_preds(predictions, prop_map):
  for i, parents in enumerate(prop_map):
    if not parents:
      continue
            
    col_filho = predictions[:, i:i+1]
    predictions[:, parents] = np.maximum(predictions[:, parents], col_filho)

  return predictions

def evaluate_wfmax(preds, gt, ontologies_names, ontology, ic, prop_map):
  preds = propagate_preds(preds, prop_map)
  
  wfmax = 0.0
  for tau in np.linspace(0.01, 1.0, 10):
    wpr, wrc, num_prot_w = 0, 0, 0
    for i, pred_row in enumerate(preds):
      pred_indices = np.where(pred_row >= tau)[0]
      gt_indices = np.where(gt[i] == 1)[0]
      
      protein_pred = [ontologies_names[k] for k in pred_indices]
      protein_gt = [ontologies_names[k] for k in gt_indices]
      
      intersect = set(protein_pred).intersection(set(protein_gt))
      
      ic_pred = sum(ic.get(q, 0) for q in protein_pred)
      ic_gt = sum(ic.get(q, 0) for q in protein_gt)
      ic_intersect = sum(ic.get(q, 0) for q in intersect)
      
      if ic_pred > 0:
        num_prot_w += 1
        wpr += (ic_intersect / ic_pred)
      if ic_gt > 0:
        wrc += (ic_intersect / ic_gt)
        
    tau_wpr = wpr / num_prot_w if num_prot_w > 0 else 0
    tau_wrc = wrc / len(preds)
    
    if tau_wrc + tau_wpr > 0:
      tau_wfmax = (2 * tau_wpr * tau_wrc) / (tau_wpr + tau_wrc)
      wfmax = max(wfmax, tau_wfmax)
      
  return wfmax