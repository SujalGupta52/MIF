lr=1e-3
epoch=100
shots=16
seed=1
delta=0.01
gammar=0.3
backbone="RN50" # RN50 RN101 ViT-B/32 ViT-B/16

# GPU ID
gpuid=1

# 数据集列表（包括 ImageNet 的情况）
datasets=("imagenet" "caltech101" "dtd" "eurosat" "fgvc" "food101" "oxford_flowers" "oxford_pets" "stanford_cars" "sun397" "ucf101")

for dataset in "${datasets[@]}"
do
    if [ "$dataset" == "imagenet" ]; then
        gpuid=1
        title="FAR_lr${lr}"
        log_file="${title}.log"

        CUDA_VISIBLE_DEVICES=${gpuid} python train_imagenet.py \
        --config ./configs/imagenet.yaml \
        --backbone ${backbone} \
        --shots ${shots} \
        --seed ${seed} \
        --delta ${delta} \
        --gammar ${gammar} \
        --train_epoch ${epoch} --lr ${lr} \
        --title ${title} --log_file ${log_file} \
        --desc "GPU${gpuid}, nKL = ${delta}, L1 = ${gammar}"

    else
        title="${dataset}_FAR_lr${lr}"
        log_file="${title}.log"

        CUDA_VISIBLE_DEVICES=${gpuid} python train_datasets.py \
        --config ./configs/${dataset}.yaml \
        --backbone ${backbone} \
        --shots ${shots} \
        --seed ${seed} \
        --delta ${delta} \
        --gammar ${gammar} \
        --train_epoch ${epoch} --lr ${lr} \
        --title ${title} --log_file ${log_file} \
        --desc "GPU${gpuid}, nKL = ${delta}, L1 = ${gammar}"
    fi
done