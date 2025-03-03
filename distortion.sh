#!/bin/bash
#SBATCH --job-name=all_distortion_fig_6
#SBATCH --qos=normal
#SBATCH --partition=rtx4090
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

cd $HOME/dgm-eval
export PATH=$HOME/miniconda3/bin:$PATH
source activate dgm-eval

REPS_TRAIN_PATH="/shared/sets/datasets/CIFAR10-dgm_eval_reps/dinov2/train"
REPS_TEST_PATH="/shared/sets/datasets/CIFAR10-dgm_eval_reps/dinov2/test"
DATASET_BASE="/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-PFGMPP/interpolation"
OUTPUT_BASE="$HOME/dgm-eval/distortions"

MODEL="PFGMPP"
DISTORTIONS=("none" "posterize" "blur" "resize" "center_crop30" "center_crop28" "color_distort" "elastic_transform" "jpg75" "jpg90")
METRICS=("authpct" "fd" "cmmd" "fld")
for train_folder in "$DATASET_BASE"/*train; do
    TRAIN_NAME=$(basename "$train_folder")
    for metric in "${METRICS[@]}"; do
        for distortion in "${DISTORTIONS[@]}"; do
            OUTPUT_DIR="$OUTPUT_BASE/$MODEL/$metric/$TRAIN_NAME/$distortion"
            mkdir -p "$OUTPUT_DIR"
            echo "Applying distortion: $distortion to folder: $train_folder"
            
            python -m dgm_eval "$REPS_TRAIN_PATH" "$train_folder" \
                --test_path "$REPS_TEST_PATH" \
                --output_dir "$OUTPUT_DIR" \
                --model dinov2 \
                --metrics "$metric" \
                --device cuda \
                --batch_size 256 \
                --nsample 10000 \
                --no-load \
                --use_test_train_repr \
                --eval_feat gap \
                --distortion "$distortion"
        
            echo "Completed evaluation for model $MODEL with distortion $distortion for metric $metric"
        done
    done
done
