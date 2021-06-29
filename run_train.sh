#!/bin/bash
# matlab -nodisplay -nosplash -nodesktop -r "run('/home/xiaohui8/Desktop/tube_samples_dataset/GoogLeNet/googlenet_pretrain.m');exit;"|tail -n +11
img_size=224
net_layer=50
python train.py --model_name=resnet${net_layer} \
                --image_dir="/Users/zongfan/Projects/data/covidx_test" \
                --image_size=$img_size \
                --num_classes=3 \
                --batch_size=16 \
                --num_epochs=100 \
                --model_save_path="test" \
                --device="cpu" \
                --lr=0.001 \
                --moment=0.9 \
                --use_pretrained=True \
                --loss="cross-entropy" \
                --dataset="test" \
                # --balanced_sampling="balanced" \
                # --num_samples=20000