#!/bin/bash

#SBATCH --job-name alignment 	 # good manners rule

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


### init virtual environment
source /BS/nshvetso/work/miniconda3/bin/activate ""
conda activate imp_videocap3

export PYTHONPATH=$PYTHONPATH:"./"

#config=blip_debug_s50
#llm_predictions=final_prompt_s50
#python howtocaption/align_and_filter.py \
#  --frame_embeddings output/embeddings/video_${config}.pickle \
#  --text_embeddings output/embeddings/text_${config}_${llm_predictions}.pickle \
#  --top_pairs_threshold 600 \
#  --output output/generated_dataset/${config}_${llm_predictions}.pickle

config1=blip_debug_s50
config2=blip_ft_1round_s50
llm_predictions=final_prompt_s50
python howtocaption/align_and_filter.py \
  --frame_embeddings output/embeddings/video_${config1}.pickle output/embeddings/video_${config2}.pickle \
  --text_embeddings output/embeddings/text_${config1}_${llm_predictions}.pickle  output/embeddings/text_${config2}_${llm_predictions}.pickle \
  --top_pairs_threshold 600 \
  --output output/generated_dataset/average_${llm_predictions}.pickle

#  --output_root output/generated_dataset/${config}_${llm_predictions}_tmp/


#python imp_videocap/create_alignment.py --dataset ${llm_model}_${exp}_${subset}_filtered --subset ${subset} --exclusive 0 \
#      --threshold_top_pairs ${pairs} --features ${predict_config} --apply_offset_to_the_middle_of_the_segment 1 \
#      --process_only_part_i $process_only_part_i --number_of_parts $number_of_parts &



