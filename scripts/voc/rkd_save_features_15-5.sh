#!/bin/bash

set -e

start=`date +%s`

START_DATE=$(date '+%Y-%m-%d')

PORT=$((9000 + RANDOM % 1000))
GPU=1
NB_GPU=1


DATA_ROOT=/media/Z/data/PascalVOC12

DATASET=voc
TASK=15-5
LOSS=0.5
NAME=RKD
METHOD=PLOP

OPTIONS="--checkpoint checkpoints/step/"

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

BATCH_SIZE=4
INITIAL_EPOCHS=30
EPOCHS=30
FIRSTMODEL=checkpoints/step/15-5s-voc_RKD_0.pth

CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} save_features.py --ckpt ${FIRSTMODEL} --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 0 --lr 0.01 --epochs ${INITIAL_EPOCHS} --method ${METHOD} --ClassSimilarityWeightedKD ${LOSS} --opt_level O1 ${OPTIONS}
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} save_features.py --ckpt checkpoints/step/voc/15-5-voc_RKD_1.pth --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 1 --lr 0.001 --epochs ${EPOCHS} --method ${METHOD} --ClassSimilarityWeightedKD ${LOSS} --opt_level O1 ${OPTIONS}

echo ${SCREENNAME}


end=`date +%s`
runtime=$((end-start))
echo "Run in ${runtime}s"
