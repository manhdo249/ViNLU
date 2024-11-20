#!/bin/bash

# Parse command-line arguments for pretrained_path
while getopts p: flag
do
    case "${flag}" in
        p) pretrained_path=${OPTARG};;
    esac
done

# Check if pretrained_path was provided
if [ -z "$pretrained_path" ]
then
    echo "Error: Pretrained path is not provided. Use -p <path> to specify it."
    exit 1
fi

# Train JointIDSF
export lr=4e-5
export c=0.4
export s=100
echo "${lr}"
export MODEL_DIR=JointIDSF_PhoBERTencoder
export MODEL_DIR=$MODEL_DIR"/"$lr"/"$c"/"$s
echo "${MODEL_DIR}"

python3 main.py --token_level word-level \
                  --model_type phobert \
                  --model_dir $MODEL_DIR \
                  --data_dir PhoATIS \
                  --seed $s \
                  --do_train \
                  --do_eval \
                  --save_steps 140 \
                  --logging_steps 140 \
                  --num_train_epochs 50 \
                  --tuning_metric mean_intent_slot \
                  --use_intent_context_attention \
                  --attention_embedding_size 200 \
                  --use_crf \
                  --gpu_id 0 \
                  --embedding_type soft \
                  --intent_loss_coef $c \
                  --pretrained \
                  --pretrained_path $pretrained_path \
                  --learning_rate $lr
