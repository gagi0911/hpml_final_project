#!/bin/bash
#SBATCH --job-name=0vit_model_rtx_4_dmodel_300_batch_512_cifar10
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256GB
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --time=15:00:00
#SBATCH --output="0vit_model_rtx_4_dmodel_300_batch_512_cifar10.txt"
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=hl5035@nyu.edu

module purge
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
# export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
# echo "WORLD_SIZE="$WORLD_SIZE

echo "MASTER_PORT="$MASTER_PORT
export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"
echo "MASTER_ADDR="$MASTER_ADDR


source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh
conda activate /scratch/hl5035/hpml


export PYTHONUNBUFFERED=TRUE
python vit.py --batch 512 --epochs 350 --gpu 2 --dmodel 300 --out 0vit_model_rtx_4_dmodel_300_batch_512_cifar10
