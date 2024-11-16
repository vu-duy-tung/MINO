#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export WORLD_SIZE=4
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=15000
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
export PYTHONPATH="$PYTHONPATH:$PWD"
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="/home/admin/miniconda3/envs/graphui2code/lib"

torchrun --standalone --nproc_per_node=gpu eval/inference.py --pth_to_ckpt ./pth/to/model/ckpt