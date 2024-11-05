lr=1e-3
shots_values=(1 2 4 8 16)  # Define an array of shots values
seed=1
delta=0.01
gammar=0.3
#backbones=("ViT-B/16" "ViT-B/32" "RN50" "RN101")  # Add backbones into an array
backbones=("RN50" "ViT-B/16")  # Add backbones into an array

# GPU ID
gpuid=0

# 数据集列表（包括 ImageNet 的情况）
datasets=("imagenet" "caltech101" "dtd" "eurosat" "fgvc" "food101" "oxford_flowers" "oxford_pets" "stanford_cars" "sun397" "ucf101")

# Outer loop to iterate over each backbone
for backbone in "${backbones[@]}"
do
    # 替换斜杠 / 为下划线 _，生成安全的 backbone 名称
    safe_backbone=$(echo ${backbone} | sed 's/\//_/g')

    # Inner loop to iterate over each shots value
    for shots in "${shots_values[@]}"
    do
        # 根据 shots 和数据集类型调整
        for dataset in "${datasets[@]}"
        do
            if [ "$dataset" == "imagenet" ]; then
                gpuid=0

                title="imagenet_FAR_${safe_backbone}_lr${lr}_shots${shots}"
                log_file="${title}.log"

                CUDA_VISIBLE_DEVICES=${gpuid} python train_imagenet.py \
                --config ./configs/imagenet.yaml \
                --backbone ${backbone} \
                --shots ${shots} \
                --seed ${seed} \
                --delta ${delta} \
                --gammar ${gammar} \
                --lr ${lr} \
                --title ${title} --log_file ${log_file} \
                --desc "GPU${gpuid}, Backbone = ${backbone}, nKL = ${delta}, L1 = ${gammar}"

            else
                title="${dataset}_FAR_${safe_backbone}_lr${lr}_shots${shots}"
                log_file="${title}.log"

                CUDA_VISIBLE_DEVICES=${gpuid} python train_datasets.py \
                --config ./configs/${dataset}.yaml \
                --backbone ${backbone} \
                --shots ${shots} \
                --seed ${seed} \
                --delta ${delta} \
                --gammar ${gammar} \
                --lr ${lr} \
                --title ${title} --log_file ${log_file} \
                --desc "GPU${gpuid}, Backbone = ${backbone}, nKL = ${delta}, L1 = ${gammar}"
            fi
        done
    done
done