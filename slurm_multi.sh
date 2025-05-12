#!/bin/bash
#SBATCH --job-name=verl_rloo
#SBATCH --array=0-4
#SBATCH -o ./slurm/%x/job_%A_%a.out # STDOUT
# SBATCH -p HGPU
#SBATCH --gres=gpu:H100:1    # Request N GPUs per machine
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 4
#SBATCH --chdir /home/zhaoyiz/projects/reasoning/verl

# script=examples/ppo_trainer/verify_qwen2.5-math-1.5b_gsm8k.sh
script=examples/rloo_trainer/run_qwen2.5_math_1.5b_MATH.sh
# script=examples/ppo_trainer/verify_deepseek-1.5b_gsm8k.sh

export task_id=${SLURM_ARRAY_TASK_ID}
local_dir=.local_verl
# local_dir=.local_reason
apptainer exec --nv -B $HOME/$local_dir:$HOME/.local --env-file  $HOME/containers/env.txt $HOME/containers/pytorch_23.11-py3.sif /bin/bash $script
