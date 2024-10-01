#!/bin/bash

#SBATCH --job-name save_text_embeddings 	 # good manners rule

#SBATCH --partition	gpu22
#SBATCH --exclude=gpu22-a40-02

#SBATCH --time=24:00:00	# hh:mm:ss, walltime (less requested time -> less time in queue)

#SBATCH --gres=gpu:1 # number of GPUs to use (Per node!) Max 4 per node
#SBATCH --nodes=1		# amount of nodes allocated
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=125Gb

#SBATCH -o /u/nshvetso/logs/improving_video_captioning/%x-%j-%N.out # STDOUT
#SBATCH -e /u/nshvetso/logs/improving_video_captioning/%x-%j-%N.err # STDERR

set -ex

export MASTER_PORT=12356
export WORLD_SIZE=1

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export PYTHONPATH=$PYTHONPATH:./

### init virtual environment
source /BS/nshvetso/work/miniconda3/bin/activate ""
conda activate imp_videocap3

#config=blip_debug_s50
#llm_predictions=final_prompt_s50
#python howtocaption/save_text_embeddings.py \
#    --llm_predictions output/vicuna/${llm_predictions}.pickle  \
#    -c configs/align_and_filter/${config}.yaml


config=blip_ft_1round_s50
llm_predictions=final_prompt_s50
python howtocaption/save_text_embeddings.py \
    --llm_predictions output/vicuna/${llm_predictions}.pickle  \
    -c configs_additional/${config}.yaml


#python howtocaption/save_text_embeddings.py \
#    --llm_predictions output/vicuna/final_prompt_s50.pickle  \
#    -c configs/align_and_filter/blip.yaml \
#    --csv data/howto100m/video_path_filtered_50.cvs