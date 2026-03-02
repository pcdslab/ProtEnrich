#!/bin/bash 
#SBATCH --job-name=protenrich
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=8 
#SBATCH --gpus=1
#SBATCH --nodelist=dragon-compute-06
#SBATCH --mem-per-cpu=2gb 
#SBATCH --output=%x-%j.out 
#SBATCH --error=%x-%j.err 

echo "Job started at: $(date)" 
echo "Running on node: $(hostname)" 
echo "Current working directory: $(pwd)" 
echo "Job ID: ${SLURM_JOB_ID}" 
echo "Job Name: ${SLURM_JOB_NAME}" 

source ~/miniconda3/etc/profile.d/conda.sh 
conda activate protenrich1

cd src
# python extract_pdb.py
# python extract_data.py --model_name structure
# python extract_data.py --model_name dynamics
# python extract_data.py --model_name protbert
# python train.py --model_name protbert
# python downstream.py --task fluorescence --model_name protbert
# python retrieval.py --task family --model_name protbert
# python extract_pdb.py --dataset_name out-of-distribution
# python extract_data.py --model_name protbert --dataset_name out-of-distribution
# python generate.py --task out-of-distribution --model_name protbert

conda deactivate

echo "Job Finished at: $(date)"