#!/usr/bin bash
do
    python MOGB.py \
    --dataset   banking \
    --known_cls_ratio 0.75 \
    --labeled_ratio 1.0 \
    --seed 0\
    --freeze_bert_parameters \
    --gpu_id 0 \
    --train_batch_size 128 \
    --eval_batch_size 2048 \
done