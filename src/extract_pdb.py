import requests
import os
import concurrent.futures
from tqdm import tqdm
from datasets import load_dataset 
import click

OUTPUT_DIR = "../pdb_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MAX_WORKERS = 8
MODEL_VERSION = "v6"
DATASETS = ['pretraining', 'out-of-distribution']

def download_pdb(uniprot_id):
  uniprot_id = uniprot_id.strip()
  file_name = f"AF-{uniprot_id}-F1-model_{MODEL_VERSION}.pdb"
  file_path = os.path.join(OUTPUT_DIR, file_name)
  url = f"https://alphafold.ebi.ac.uk/files/{file_name}"

  if os.path.exists(file_path):
    return uniprot_id, "already_exists"

  try:
    response = requests.get(url, timeout=60)
    if response.status_code == 200:
      with open(file_path, 'wb') as f:
        f.write(response.content)
      return uniprot_id, "success"
    else:
      return uniprot_id, f"failed_status_{response.status_code}"

  except requests.exceptions.RequestException as e:
    return uniprot_id, f"error_{e}"

@click.command()
@click.option('--dataset_name', 
  required=True,
  type=click.Choice(DATASETS),
  help="Dataset name - pretraining or out-of-distribution.",
  default="pretraining"
)
def main(dataset_name):
  if dataset_name == 'pretraining':
    dataset = load_dataset("SaeedLab/ProtEnrich", data_dir="pretraining")['train']['prot_id'][:10]
  elif dataset_name == 'out-of-distribution':
    dataset = load_dataset("SaeedLab/ProtEnrich", data_dir="out-of-distribution")['train'][:10]
  else:
    raise ValueError(f"Invalid dataset: {dataset_name}")

  with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    results = list(tqdm(executor.map(download_pdb, dataset), total=len(dataset), desc="Downloading PDBs"))

  success_count = sum(1 for _, status in results if status.startswith("success"))
  exists_count = sum(1 for _, status in results if status == "already_exists")
  failed_ids = [(uid, status) for uid, status in results if not status.startswith("success") and status != "already_exists"]

  print("\n--- Summary ---")
  print(f"Total number of IDs: {len(dataset)}")
  print(f"Successfully: {success_count}")
  print(f"Already exists: {exists_count}")
  print(f"Error: {len(failed_ids)}")

  if failed_ids:
    print("\nSaving failed ids in 'failed_ids.txt'...")
    with open("failed_ids.txt", "w") as f:
      for uid, status in failed_ids:
        f.write(f"{uid.strip()}\t{status}\n")

if __name__ == "__main__":
  main()
