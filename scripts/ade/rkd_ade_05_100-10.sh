#!/bin/bash

set -e

start=`date +%s`

START_DATE=$(date '+%Y-%m-%d')

PORT=$((9000 + RANDOM % 1000))
GPU=1
NB_GPU=1

DATA_ROOT=/media/hieu/DATA/semantic_segmentation

DATASET=ade
TASK=100-10
BATCH_SIZE=10
NAME=RKD_TEST_80
LOSS=0.5
EPOCHS=80
LR=0.0008
METHOD=FT
OPTIONS="--checkpoint checkpoints/ --pod local --pod_factor 0.001 --pod_logits --pseudo entropy --threshold 0.001 --classif_adaptive_factor --init_balanced --csw_kd ${LOSS}"
FIRSTMODEL=checkpoints/ade/100-10-ade_RKD_0.pth

SCREENNAME="${DATASET}_${TASK}_${NAME} On GPUs ${GPU}"

RESULTSFILE=results/${START_DATE}_${DATASET}_${TASK}_${NAME}.csv
rm -f ${RESULTSFILE}

echo -ne "\ek${SCREENNAME}\e\\"

echo "Writing in ${RESULTSFILE}"

# If you already trained the model for the first step, you can re-use those weights
# in order to skip this initial step --> faster iteration on your model
# Set this variable with the weights path
# FIRSTMODEL=/path/to/my/first/weights
# Then, for the first step, append those options:
# --ckpt ${FIRSTMODEL} --test
# And for the second step, this option:
# --step_ckpt ${FIRSTMODEL}

CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --ckpt ${FIRSTMODEL} --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 0 --lr 0.01 --epochs 60 --method ${METHOD} --opt_level O1 ${OPTIONS}
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --step_ckpt ${FIRSTMODEL} --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 1 --lr ${LR} --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} --pod_options "{\"switch\": {\"after\": {\"extra_channels\": \"sum\", \"factor\": 0.00001, \"type\": \"local\"}}}"
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 2 --lr ${LR} --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} --pod_options "{\"switch\": {\"after\": {\"extra_channels\": \"sum\", \"factor\": 0.00001, \"type\": \"local\"}}}"
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 3 --lr ${LR} --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} --pod_options "{\"switch\": {\"after\": {\"extra_channels\": \"sum\", \"factor\": 0.00001, \"type\": \"local\"}}}"
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 4 --lr ${LR} --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} --pod_options "{\"switch\": {\"after\": {\"extra_channels\": \"sum\", \"factor\": 0.00001, \"type\": \"local\"}}}"
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 5 --lr ${LR} --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} --pod_options "{\"switch\": {\"after\": {\"extra_channels\": \"sum\", \"factor\": 0.00001, \"type\": \"local\"}}}"
python3 average_csv.py ${RESULTSFILE}

echo ${SCREENNAME}


end=`date +%s`
runtime=$((end-start))
echo "Run in ${runtime}s"