#!/bin/bash
# model directory
model_dir="/shared/anastasio5/COVID19/covid19_classification_with_gan.old/covidx_gan_res50_patch_224"
# model names for testing
declare -a StringArray=("cls_classifier_epoch_15.pt" "cls_classifier_epoch_18.pt" "cls_classifier_epoch_21.pt" "cls_classifier_epoch_24.pt" "cls_classifier_epoch_27.pt" "best_model.pt")
image_dir="/shared/anastasio5/COVID19/data/covidx"
# for debugging
# model_dir="test"
# declare -a StringArray=("Unet_epoch_10.pt")
# imamge_dir=/Users/zongfan/Projects/data/covidx_test
for model in ${StringArray[@]};
do
    full_path="$model_dir/$model"
    python eval.py --model_name="resnet50" \
               --num_classes=3 \
               --model_weights=$full_path \
               --image_dir=$image_dir \
               --image_size=224 \
               --device="cpu" \
               --dataset="covidx" \
               --is_unet=False \
               --multi_gpus=True \
               --auc=True
    echo "Model processed: $model"
    echo "======================="
done 
