lr=1e-3
epoch=100
shots=(1 2 4 8 16)  # Define shots as an array
seed=1
delta=0.01
gammar=0.3
backbone="RN50"  # Options: RN50, RN101, ViT-B/32, ViT-B/16

gpuid=0

# 数据集列表
datasets=("caltech101" "dtd" "eurosat" "fgvc" "food101" "oxford_flowers" "oxford_pets" "stanford_cars" "sun397" "ucf101")

for num_shots in "${shots[@]}"  # Loop through each value in the shots array
do
    for dataset in "${datasets[@]}"
    do
        title="${dataset}_FAR_lr${lr}_shots${num_shots}"  # Include shots value in title
        log_file="${title}.log"

        CUDA_VISIBLE_DEVICES=${gpuid} python train_datasets.py \
        --config ./configs/${dataset}.yaml \
        --backbone ${backbone} \
        --shots ${num_shots} \  # Use num_shots from the loop
        --seed ${seed} \
        --delta ${delta} \
        --gammar ${gammar} \
        --train_epoch ${epoch} --lr ${lr} \
        --title ${title} --log_file ${log_file} \
        --desc "GPU${gpuid}, nKL = ${delta}, L1 = ${gammar}"
    done
done