lr=1e-3
epoch=100
seed=1
delta=0.01
gammar=0.3
gpuid=1

# 定义可选的 backbone 列表
#backbone_list=("RN50" "RN101" "ViT-B/32" "ViT-B/16")
backbone_list=("RN50" "ViT-B/16")

# 定义 shots 列表
shots_list=(1 2 4 8 16)

# 循环遍历 backbone 和 shots 列表
for backbone in "${backbone_list[@]}"; do
    for shots in "${shots_list[@]}"; do
        title=FAR_lr${lr}_shots${shots}_backbone${backbone}
        log_file=${title}.log

        CUDA_VISIBLE_DEVICES=${gpuid} python train_imagenet.py \
        --config ./configs/imagenet.yaml \
        --backbone ${backbone} \
        --shots ${shots} \
        --seed ${seed} \
        --delta ${delta} \
        --gammar ${gammar} \
        --train_epoch ${epoch} --lr ${lr} \
        --title ${title} --log_file ${log_file} \
        --desc "GPU${gpuid}, nKL = ${delta}, L1 = ${gammar}, Shots = ${shots}, Backbone = ${backbone}"
    done
done