lr=1e-3
shots=16
seed=1
delta=0.01
gammar=0.5
backbone="RN50" # RN50 RN101 ViT-B/32 ViT-B/16          1
#backbone="ViT-B/32" # RN50 RN101 ViT-B/32 ViT-B/16
#backbone="RN101" # RN50 RN101 ViT-B/32 ViT-B/16
#backbone="ViT-B/16" # RN50 RN101 ViT-B/32 ViT-B/16

gpuid=1

title=FAR_lr${lr}
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
 --desc "GPU${gpuid}, nKL = ${delta}, L1 = ${gammar}" \
