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
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_LIBRARY_PATH="/home/admin/miniconda3/envs/graphui2code/lib"

torchrun --standalone --nproc_per_node=gpu train/finetuning.py \
    --lm_path anas-awadalla/mpt-1b-redpajama-200b \
    --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
    --cross_attn_every_n_layers 4 \
    --dataset_resampled \
    --batch_size_websight 16 \
    --train_num_samples_websight 20000 \
    --workers=4 \
    --run_name OpenFlamingo-2.5B-vitl-mpt1b-no-graph \
    --num_epochs 100 \
    --warmup_steps  1875 \
    --logging_steps 100 \
    --save_ckpt_steps 2000\
    --learning_rate 1e-6 \
    --wandb_project 27-08-2024 \
    --gradient_checkpointing \
    --delete_previous_checkpoint \
    --report_to_wandb


