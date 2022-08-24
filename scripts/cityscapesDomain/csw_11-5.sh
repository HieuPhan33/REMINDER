#!/bin/bash

set -e

start=`date +%s`

START_DATE=$(date '+%Y-%m-%d')

PORT=$((9000 + RANDOM % 1000))
GPU=0,1
NB_GPU=2


DATA_ROOT=/media/Z/data/cityscapes

DATASET=cityscapes_domain
TASK=10-5
NAME=REMINDER
METHOD=FT
LOSS=0.6
OPTIONS="--checkpoint checkpoints/step/ --pod local --pod_factor 0.0001 --pod_logits"
NB_EPOCHS=50
FIRSTMODEL=checkpoints/step/cityscapes_domain/10-5-cityscapes_domain_REMINDER_0.pth

SCREENNAME="${DATASET}_${TASK}_${NAME} On GPUs ${GPU}"

RESULTSFILE=results/${START_DATE}_${DATASET}_${TASK}_${NAME}.csv
rm -f ${RESULTSFILE}

echo -ne "\ek${SCREENNAME}\e\\"

echo "Writing in ${RESULTSFILE}"

#CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --ckpt ${FIRSTMODEL} --ClassSimilarityWeightedKD ${LOSS} --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size 12 --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 0 --lr 0.01 --epochs ${NB_EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS}
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --ckpt checkpoints/step/cityscapes_domain/10-5-cityscapes_domain_REMINDER_1.pth --ClassSimilarityWeightedKD ${LOSS} --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size 12 --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 1 --lr 0.001 --epochs ${NB_EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} --pod_options "{\"switch\": {\"after\": {\"extra_channels\": \"sum\", \"factor\": 0.0001, \"type\": \"local\"}}}"
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --ClassSimilarityWeightedKD ${LOSS} --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size 12 --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 2 --lr 0.001 --epochs ${NB_EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} --pod_options "{\"switch\": {\"after\": {\"extra_channels\": \"sum\", \"factor\": 0.0001, \"type\": \"local\"}}}"
python3 average_csv.py ${RESULTSFILE}

echo ${SCREENNAME}


end=`date +%s`
runtime=$((end-start))
echo "Run in ${runtime}s"
