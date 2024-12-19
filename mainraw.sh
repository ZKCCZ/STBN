export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
python CRVD_train_new.py \
    --model BR_d_c_raw \
    --lr 1e-4 \
    --epochs 200 \
    --batch_size 16 \
    --patch_size 80 \
    --patches_per_epoch 32000 \
    --num_resblocks 5 \
    --print_every 500 \
    --milestones 2000 \
    --gamma 0.1 \
    --log_dir ../logs/iso  \

