#!/bin/bash

#SBATCH --job-name train_retrieval_723_1200k_realign_720_all_mod_40ep_at6 # good manners rule
#SBATCH --time=11:59:00	# hh:mm:ss, walltime (less requested time -> less time in queue)
##SBATCH -a 1-4%1

#SBATCH --partition	gpu22
#SBATCH --exclude=gpu22-a40-02,gpu20-26,gpu20-23,gpu20-07,gpu20-19,gpu20-03,gpu20-06

#SBATCH --nodes=1		# amount of nodes allocated
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16

#SBATCH --mem=900Gb

#SBATCH -o /BS/nshvetso/work/logs/%x-%j-%N.out # STDOUT
#SBATCH -e /BS/nshvetso/work/logs/%x-%j-%N.err # STDERR

set -ex

export MASTER_PORT=1$((1 + $RANDOM % 10000))
export WORLD_SIZE=4

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment
source /BS/nshvetso/work/miniconda3/bin/activate
conda activate imp_videocap3

#source /BS/nshvetso/work/miniconda3_new2/bin/activate
#conda activate howtocaption

export PYTHONPATH=$PYTHONPATH:"./"

#exp=train_retrieval_723_1200k_realign_720_all_mod_40ep_at6
##exp='debug_run'
#last_exp=$(ls output/models/${exp}/*/latest_model.pth |  sort | tail -n 1)
#if [[ ${last_exp} -eq '' ]]
#then
#  srun python howtocaption/train.py \
#    -c configs/VL_training/$exp.yaml \
#    --distributed 1 \
#    --world_size ${WORLD_SIZE} \
#    --neptune
#else
#  srun python howtocaption/train.py \
#    --resume $last_exp \
#    -c configs/VL_training/${exp}.yaml \
#    --distributed 1 \
#    --world_size ${WORLD_SIZE} \
#    --neptune
#fi

#export MASTER_PORT=12345
#export WORLD_SIZE=4
#export MASTER_ADDR=localhost
#
#torchrun --standalone --nnodes=1 --nproc_per_node=${WORLD_SIZE} \
#    howtocaption/train.py \
#    -c configs/VL_training/dual_encoder_retrieval.yaml \
#    --distributed 1 \
#    --world_size ${WORLD_SIZE}

export MASTER_PORT=12345
export WORLD_SIZE=4
export MASTER_ADDR=localhost

torchrun --standalone --nnodes=1 --nproc_per_node=${WORLD_SIZE} \
    howtocaption/train.py \
    -c configs/align_and_filter/finetune_1round.yaml\
    --distributed 1 \
    --world_size ${WORLD_SIZE}

#
#srun python howtocaption/train.py \
#    -c configs/VL_training/captioning_msrvtt.yaml \
#    --resume pretrained/dual_encoder_retrieval.pth \
#    --distributed 1 \
#    --world_size ${WORLD_SIZE} \
#    --neptune

