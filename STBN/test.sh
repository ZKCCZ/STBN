eval "$(conda shell.bash hook)"
conda activate flornn
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=7

cd test_models
python sRGB_test.py \
    --model bdc_2_unify \
    --num_resblocks 5 \
    --noise_sigmas 30 \
    --model_file ../logs/pretrained.pth \
    --test_path /data/czk/Video/final/datasets/Videos/Set8
