#!/bin/bash
#SBATCH --job-name=cmmd_fig_6_duplication
#SBATCH --qos=normal
#SBATCH --partition=rtx4090
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

cd $HOME/dgm-eval
export PATH=$HOME/miniconda3/bin:$PATH
source activate dgm-eval

echo 'Running script'

REPS_TRAIN_PATH="/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10/train"
REPS_TEST_PATH="/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10/test"

REPS_TRAIN_PATH="/shared/sets/datasets/CIFAR10-dgm_eval_reps/train"
REPS_TEST_PATH="/shared/sets/datasets/CIFAR10-dgm_eval_reps/test"
DATASET_BASE="/shared/sets/datasets/PALATE_FIG_6_DUPLICATION"

MODELS=("StyleGAN-XL" "PFGMPP" "MHGAN" "StyleGAN2-ada", "BigGAN-Deep")
SUBSETS=("100" "200" "500" "1000")  # Subsets to process
METRICS=("vendi")
for model in "${MODELS[@]}"; do
    MODEL_PATH="$DATASET_BASE/CIFAR10-$model"
    for metric in "${METRICS[@]}"; do
	OUTPUT_BASE="$HOME/dgm-eval/new_cls_id_copying_all_classes_$metric"
	    for subset in "${SUBSETS[@]}"; do
		REAL_GEN_PATH="$MODEL_PATH/$subset"
		if [ -d "$REAL_GEN_PATH" ]; then
		    OUTPUT_DIR="$OUTPUT_BASE/$model/$subset"
		    mkdir -p "$OUTPUT_DIR"
		    echo "Processing folder: $REAL_GEN_PATH for subset $subset"
		    
		    python -m dgm_eval "$REPS_TRAIN_PATH" "$REAL_GEN_PATH" \
			--test_path "$REPS_TEST_PATH" \
			--output_dir "$OUTPUT_DIR" \
			--model dinov2 \
			--metrics "$metric" \
			--device cuda \
			--batch_size 256 \
			--nsample 10000 \
			--no-load \
			--fig_6
		    
		    echo "Completed evaluation for subset $subset in model $model"
		else
		    echo "Skipping $REAL_GEN_PATH as it does not exist."
		fi
	done
    done
done

echo 'Task completed'
