#!/bin/bash
  
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64GB
#SBATCH --job-name=ganidhu_fluid_sim
#SBATCH --account=st-pai-1-gpu
#SBATCH --mail-user=ganidhu@student.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/st-pai-1/ganidhu/fluid_sim/outputs/slurm_out/slurm_%j.out
#SBATCH --error=/scratch/st-pai-1/ganidhu/fluid_sim/outputs/slurm_out/slurm_%j.err
  
################################################################################

# load modules
module load gcc/5.5.0 gcc/7.5.0 gcc/9.4.0
module load http_proxy
module load miniconda3

# activate conda env
source ~/.bashrc
conda activate /arc/home/ganidhu/venvs/fluid_sim

cd /scratch/st-pai-1/ganidhu/fluid_sim/
python train.py