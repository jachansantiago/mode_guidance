
nvidia-smi

export CUDA_VISIBLE_DEVICES=0
IMAGENET_FOLDER=/mnt/ImageNet


IPC=10
T_SG=25 # STOP GUIDANCE
SPEC=imagenet100

for ((i=0; i < 3; i++))
do

OUTPUT_DATASET=results/dit-distillation-sampling-t-$T_SG/nette-$i-IPC-$IPC
TRAIN_SAVE_DIR=results/train-nette-mode-sampling-t-$T_SG/nette-$i-IPC-$IPC

python sample_mode_guidance.py --model DiT-XL/2 --image-size 256 \
 --save-dir $OUTPUT_DATASET --spec $SPEC --num-samples $IPC --guidance \
 --stop_t $STOP_T $T_SG --imagenet_dir $IMAGENET_FOLDER --seed $i --num-datasets 1

python train.py -d imagenet --imagenet_dir $OUTPUT_DATASET/dataset_0 $IMAGENET_FOLDER \
    -n convnet --depth 6 --nclass 10 --norm_type instance --ipc $IPC --tag test --slct_type random --spec $SPEC --repeat 1 \
    --save-dir $TRAIN_SAVE_DIR-convnet

python train.py -d imagenet --imagenet_dir $OUTPUT_DATASET/dataset_0 $IMAGENET_FOLDER \
    -n resnet_ap --nclass 10 --norm_type instance --ipc $IPC --tag test --slct_type random --spec $SPEC --repeat 1 \
    --save-dir $TRAIN_SAVE_DIR-resnet_ap

python train.py -d imagenet --imagenet_dir $OUTPUT_DATASET/dataset_0 $IMAGENET_FOLDER \
    -n resnet --depth 18 --nclass 10 --norm_type instance --ipc $IPC --tag test --slct_type random --spec $SPEC --repeat 1 \
    --save-dir $TRAIN_SAVE_DIR-resnet_18

done
