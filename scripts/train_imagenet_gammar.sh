lr=1e-3
shots=16
seed=1
delta=0.01
backbone="RN50" # Choose from: RN50, RN101, ViT-B/32, ViT-B/16
# backbone="ViT-B/32" # Uncomment to select different backbone
# backbone="RN101"
# backbone="ViT-B/16"

gpuid=0

# Loop through gammar values from 0.1 to 0.9 with step 0.1
for gammar in $(seq 0.1 0.1 0.9)
do
    title=FAR_lr${lr}_gammar${gammar}
    log_file=${title}.log

    CUDA_VISIBLE_DEVICES=${gpuid} python train_imagenet.py \
     --config ./configs/imagenet.yaml \
     --backbone ${backbone} \
     --shots ${shots} \
     --seed ${seed} \
     --delta ${delta} \
     --gammar ${gammar} \
     --lr ${lr} \
     --title ${title} --log_file ${log_file} \
     --desc "GPU${gpuid}, nKL = ${delta}, L1 = ${gammar}"
done