# ProtEnrich

ProtEnrich: Residual Multimodal Enrichment of Protein Sequence Embeddings

\[[Dataset on HuggingFace](https://huggingface.co/datasets/SaeedLab/ProtEnrich)\] | \[[Model Collection](https://huggingface.co/collections/SaeedLab/ProtEnrich)\] | \[[Cite](#citation)\]

The paper is under review.

## Abstract
Protein language models effectively capture evolutionary and functional signals from sequence data but lack explicit representation of the biophysical properties that govern protein structure and dynamics. Existing multimodal approaches attempt to integrate such physical information through direct fusion, often requiring multimodal inputs at inference time and distorting the sequence embedding space. Consequently, a fundamental challenge of how to incorporate structural and dynamical knowledge into sequence representations without disrupting their established semantic organization remains a field-of-research. We introduce ProtEnrich, a representation learning framework based on a residual multimodal enrichment paradigm. ProtEnrich decomposes sequence embeddings into two complementary latent subspaces, an anchor subspace that preserves sequence semantics, and an alignment subspace that encodes biophysical relationships. By converting multimodal information derived from ProstT5 and RocketSHP to a low-energy residual component, our approach injects physical representation while maintaining the original sequence embedding. Across eight diverse protein foundational models trained on 550,120 SwissProt proteins with AlphaFold structures, enriched embeddings improved zero-shot remote homology retrieval, increasing Precision@10 and MRR by up to 0.13 and 0.11, respectively. Downstream performance also improved on structure-dependent tasks, reducing fluorescence prediction error by up to 16% and increasing metal ion binding AUCROC by up to 2.4 points, while requiring only sequence input at inference.

## System Requirements
- A computer with Ubuntu 16.04 (or later) or CentOS 8.1 (or later).
- CUDA-enabled GPU with at least 6 GB of memory.

## Installation Guide

### Install Anaconda
[Step by Step Guide to Install Anaconda](https://docs.anaconda.com/anaconda/install/)


### Fork the Repository
- Fork this repository to your own account.
- Clone your fork to your machine.

### Create a Conda Environment
```bash
cd <repository_directory>
conda env create --file environment.yml
```

### Activate the Environment
```bash
conda activate protenrich
```

## Running the Experiments
### Pretraining
Follow the steps below to pretrain the ProtEnrich models.

1. Extract the PDB files from AlphaFoldDB.
```bash
cd src
python python extract_pdb.py
```
---

2. Generate the embeddings for structural, dynamical, and sequence representation.
```bash
cd src
python extract_data.py --model_name structure
python extract_data.py --model_name dynamics
python extract_data.py --model_name protbert
```
Arguments:
**--model_name**: Model (structure, dynamics, or protein language models). Protein language models available: `ankh3` (Ankh 3 XL), `carp_640m` (CARP 640M), `esm1b` (ESM1b), `esm2_t36` (ESM2 T36), `esmc_600m` (ESM Cambrian 600M), `progen2` (ProGen2), `protbert` (ProtBERT), and `prott5` (ProtT5).

---

3. Train ProtEnrich.
```bash
cd src
python train.py --model_name protbert
```
Arguments:
* **--model_name**: The protein language model that ProtEnrich is based on. Available models include: `ankh3` (Ankh 3 XL), `carp_640m` (CARP 640M), `esm1b` (ESM1b), `esm2_t36` (ESM2 T36), `esmc_600m` (ESM Cambrian 600M), `progen2` (ProGen2), `protbert` (ProtBERT), and `prott5` (ProtT5).

---

### Zero-shot Retrieval
To evaluate ProtEnrich on zero-shot retrieval task, use:
```bash
cd src
python retrieval.py --task family --model_name protbert
```
Arguments:
* **--task**: The retrieval task to evaluate: Options: `family`, `fold`, and `superfamily`.
* **--model_name**: The protein language model that ProtEnrich is based on. Available models include: `ankh3` (Ankh 3 XL), `carp_640m` (CARP 640M), `esm1b` (ESM1b), `esm2_t36` (ESM2 T36), `esmc_600m` (ESM Cambrian 600M), `progen2` (ProGen2), `protbert` (ProtBERT), and `prott5` (ProtT5).

### Generative
To generate structural and dynamical embeddings, use:
```bash
cd src
python extract_pdb.py --dataset_name out-of-distribution
python extract_data.py --model_name protbert --dataset_name out-of-distribution
python generate.py --task out-of-distribution --model_name protbert
```
Arguments:
* **--dataset_name**: The dataset with protein sequences used for generate structural and dynamical embeddings. Available dataset: `out-of-distribution`.
* **--model_name**: The protein language model that ProtEnrich is based on. Available models include: `ankh3` (Ankh 3 XL), `carp_640m` (CARP 640M), `esm1b` (ESM1b), `esm2_t36` (ESM2 T36), `esmc_600m` (ESM Cambrian 600M), `progen2` (ProGen2), `protbert` (ProtBERT), and `prott5` (ProtT5).
* **--task**: The generation task to evaluate: Options: `out-of-distribution`.

### Downstream
To evaluate ProtEnrich on downstream tasks, use:
```bash
cd src
python downstream.py --task fluorescence --model_name protbert
```
Arguments:
* **--task**: The generation task to evaluate: Options: `fluorescence`, `localization_bin`, `localization_multi`, `metal_ion_binding`, `ppi`, `protein_function_bp`, `protein_function_cc`, `protein_function_mf`, `solubility`, and `thermostability`.
* **--model_name**: The protein language model that ProtEnrich is based on. Available models include: `ankh3` (Ankh 3 XL), `carp_640m` (CARP 640M), `esm1b` (ESM1b), `esm2_t36` (ESM2 T36), `esmc_600m` (ESM Cambrian 600M), `progen2` (ProGen2), `protbert` (ProtBERT), and `prott5` (ProtT5).

---

## Citation
The paper is under review. As soon as it is accepted, we will update this section.

## License

This model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of this model and its derivatives, which include models trained on outputs from the model or datasets created from the model, is prohibited and requires prior approval. Downloading the model requires prior registration on Hugging Face and agreeing to the terms of use. By downloading this model, you agree not to distribute, publish or reproduce a copy of the model. If another user within your organization wishes to use the model, they must register as an individual user and agree to comply with the terms of use. Users may not attempt to re-identify the deidentified data used to develop the underlying model. If you are a commercial entity, please contact the corresponding author.

## Contact

For any additional questions or comments, contact Fahad Saeed (fsaeed@fiu.edu).

