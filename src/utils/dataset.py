import torch
from torch.utils.data import Dataset
import numpy as np

class MultiModalProteinDataset(Dataset):
  def __init__(self, indices, model_name, emb_dim, total_length):
    self.indices = indices
    self.model_name = model_name
    self.emb_dim = emb_dim
    self.total_length = total_length

    self.X_seq = np.memmap(f'../embs_pretraining/{model_name}.mmap', dtype=np.float32, mode='r', shape=(self.total_length, self.emb_dim))
    self.X_struct = np.memmap('../embs_pretraining/struct.mmap', dtype=np.float32, mode='r', shape=(self.total_length, 1024))
    self.X_dyn = np.memmap('../embs_pretraining/dyn.mmap', dtype=np.float32, mode='r', shape=(self.total_length, 20))

  def __len__(self):
    return len(self.indices)

  def __getitem__(self, idx):
    indice = self.indices[idx]
    inputs = {}
    inputs['seq'] = torch.from_numpy(np.array(self.X_seq[indice], dtype=np.float32, copy=True)).float()
    inputs['struct'] = torch.from_numpy(np.array(self.X_struct[indice], dtype=np.float32, copy=True)).float()
    inputs['dyn'] = torch.from_numpy(np.array(self.X_dyn[indice], dtype=np.float32, copy=True)).float()
    
    return inputs

class DownstreamDataset(Dataset):
  def __init__(self, names, labels, mode, task_type, emb_dim, mapping, names_b=None, mapper=None, emb_dir="embs"):
    self.names = names
    self.names_b = names_b
    self.labels = labels
    self.task_type = task_type
    self.mapper = mapper
    self.mode = mode
    self.emb_dim = emb_dim
    self.mapping = mapping

    if task_type == "multilabel":
      assert mapper is not None, "Multilabel requer mapper"
      self.num_classes = len(mapper)

    if task_type == "pair":
      assert names_b is not None, "Pair task requer names_b"

    self.X_feat = np.memmap(f"{emb_dir}/{mode}.mmap", dtype=np.float32, mode="r", shape=(len(self.mapping), emb_dim))

  def __len__(self):
    return len(self.names)

  def _get_embedding(self, name):
    idx = self.mapping[name]
    return torch.from_numpy(np.array(self.X_feat[idx], dtype=np.float32, copy=True))

  def __getitem__(self, idx):
    if self.task_type == "pair":
      vec_a = self._get_embedding(self.names[idx])
      vec_b = self._get_embedding(self.names_b[idx])
      X = torch.cat([vec_a, vec_b], dim=0)
      y = torch.tensor(self.labels[idx], dtype=torch.long)
      return X, y

    X = self._get_embedding(self.names[idx])

    if self.task_type in ["binary", "multiclass"]:
      y = torch.tensor(self.labels[idx], dtype=torch.long)

    elif self.task_type == "regression":
      y = torch.tensor(self.labels[idx], dtype=torch.float32)

    elif self.task_type == "multilabel":
      y = torch.zeros(self.num_classes, dtype=torch.float32)
      for label in self.labels[idx]:
        if label in self.mapper:
          y[self.mapper[label]] = 1.0

    else:
      raise ValueError(f"Unknown task_type: {self.task_type}")

    return X, y