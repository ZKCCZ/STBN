export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
python sRGB_train_bsn.py \
    --model bdc_2_unify \
    --lr 3e-4 \
    --epochs 10 \
    --batch_size 4 \
    --patch_size 96 \
    --patches_per_epoch 32000 \
    --num_resblocks 5 \
    --print_every 1000 \
    --train_length 8 \
    --noise_ival 30 30 \
    --val_noiseL 30 \
    --milestones 2 4 6 8 \
    --gamma 0.5 \
    --log_dir ../logs/test\
    --trainset_dir /data/DAVIS \
    --valset_dir /data/Set8 \

