export CUDA_VISIBLE_DEVICES=0
IMAGENET_FOLDER=/home/c3-0/datasets/ImageNet/
IPC=10
SPEC=nette

# for ((i=0; i < 3; i++))
# do

OUTPUT_DATASET=results/stable-diffussion-distillation-sampling/$SPEC-IPC-$IPC
TRAIN_SAVE_DIR=results/sd-train-$SPEC-mode-sampling-t/$SPEC-$i-IPC-$IPC

python sample_mode_guidance_text2img.py --image-size 512 \
 --ckpt stable-diffusion-v1-5/ \
 --save-dir $OUTPUT_DATASET --spec $SPEC --num-samples $IPC --guidance --imagenet_dir $IMAGENET_FOLDER --seed 0 --num-datasets 1

python train.py -d imagenet --imagenet_dir $OUTPUT_DATASET/dataset_0 $IMAGENET_FOLDER \
    -n resnet_ap --nclass 10 --norm_type instance --ipc $IPC --tag test --slct_type random --spec $SPEC --repeat 1 \
    --save-dir $TRAIN_SAVE_DIR-resnet_ap

