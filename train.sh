#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONWARNINGS="ignore"

# resnet18, resnet34
export NET='resnet18'
export path='model'
export data_base='fg-web-data/web-bird'
export val_base='validation_data/bird'
export N_CLASSES=200
export lr=0.01
export w_decay=1e-5

python train.py --net ${NET} --n_classes ${N_CLASSES} --path ${path} --data_base ${data_base} --validation_base ${val_base} --lr ${lr} --w_decay ${w_decay} --drop_rate 0.35 --relabel_rate 0.05  --ts 5
