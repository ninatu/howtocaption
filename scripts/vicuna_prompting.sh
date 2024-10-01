#!/bin/bash

#SBATCH --job-name 65_whisper_as52	 # good manners rule
#SBATCH -a 1-500%20

#SBATCH --partition=gpu16
#SBATCH --time=0:59:00	# hh:mm:ss, walltime (less requested time -> less time in queue)
##SBATCH --time=11:59:00	# hh:mm:ss, walltime (less requested time -> less time in queue)

#SBATCH --gres=gpu:1 # number of GPUs to use (Per node!) Max 4 per node
#SBATCH --nodes=1		# amount of nodes allocated
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=gpu22-a40-06,gpu16-a40-13

#SBATCH --mem=125Gb

#SBATCH -o /BS/nshvetso/work/logs/fastchat/%x-%j-%N.out # STDOUT
#SBATCH -e /BS/nshvetso/work/logs/fastchat/%x-%j-%N.err # STDERR

source /BS/nshvetso/work/miniconda3_new2/bin/activate
conda activate howtocaption

python howtocaption/llm_prompting/prompt_vicuna.py --config configs/vicuna/final_prompt.yaml \
  --asr-path data/howto100m/asr_filtered_s50.pickle \
  --model-path '/BS/nshvetso/work/cache/huggingface/transformers/models--vicuna-13b'



