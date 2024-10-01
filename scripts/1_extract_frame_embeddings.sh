#!/bin/bash

#SBATCH --job-name video_features_blip 	 # good manners rule

#SBATCH --partition	gpu22
#SBATCH --exclude=gpu22-a40-02

#SBATCH --time=6:00:00	# hh:mm:ss, walltime (less requested time -> less time in queue)

#SBATCH --gres=gpu:1 # number of GPUs to use (Per node!) Max 4 per node
#SBATCH --nodes=1		# amount of nodes allocated
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=125Gb

#SBATCH -o /BS/nshvetso/work/logs/%x-%j-%N.out # STDOUT
#SBATCH -e /BS/nshvetso/work/logs/%x-%j-%N.err # STDERR


export MASTER_PORT=12356
export WORLD_SIZE=1

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment
source /BS/nshvetso/work/miniconda3/bin/activate ""
conda activate imp_videocap3

export PYTHONPATH=$PYTHONPATH:"./"

#config=blip_debug_s50
#python howtocaption/save_frame_embeddings.py \
#  -c configs/align_and_filter/${config}.yaml


#config=blip
#python howtocaption/save_frame_embeddings.py \
#  -c configs/align_and_filter/${config}.yaml

config=blip_ft_1round_s50
python howtocaption/save_frame_embeddings.py \
  -c configs_additional/${config}.yaml

#config=blip
#process_only_part_i=$1
#python howtocaption/save_frame_embeddings.py \
#  -c configs/align_and_filter/${config}.yaml \
#  --process_only_part_i ${process_only_part_i} \
#  --number_of_parts 64
