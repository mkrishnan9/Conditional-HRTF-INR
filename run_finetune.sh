#!/bin/bash

# This script runs multiple training experiments (pre-training and fine-tuning).

echo "Starting hyperparameter sweep..."

# --- Hyperparameters ---
SAVE_DIR_BASE="/export/mkrishn9/hrtf_field/experiments_ft/hutubs_cipic"
LEARNING_RATES=(1e-4)
LEARNING_RATES_FT=(1e-4 5e-5) #  learning rates for fine-tuning
LATENT_DIMS=(32 64 128)
HIDDEN_DIMS=(128 256 512)
N_LAYERS=(4 5 6)
BATCH_SIZE=(32 16 8 4 1)
EPOCHS=(1000 500)
EPOCHS_FT=(200 300 400 500 600 800 1000) # Specific epoch counts for fine-tuning
GPUS=(1 3 4 5 6 7)

job_count=0
NUM_GPUS=${#GPUS[@]}



# ==============================================================================
#  PRE-TRAINING on 'chedar' dataset
# ==============================================================================
# echo "--- LAUNCHING PRE-TRAINING EXPERIMENTS ---"

for epoch in "${EPOCHS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        for ld in "${LATENT_DIMS[@]}"; do
            for hd in "${HIDDEN_DIMS[@]}"; do
                for nl in "${N_LAYERS[@]}"; do
                    for bs in "${BATCH_SIZE[@]}"; do
                      #if [[ "$epoch" -eq 1000 && ( "$lr" == "1e-4" || ( "$lr" == "5e-5" && "$ld" -eq 64 && "$hd" -eq 64 && "$nl" -eq 3 ) ) ]]; then

                        while [[ $(jobs -p | wc -l) -ge $NUM_GPUS ]]; do
                            echo "All GPUs are busy. Waiting for a job to finish..."
                            wait -n
                        done

                        # Assign a GPU from the list
                        gpu_index=$((job_count % NUM_GPUS))
                        GPU_ID=${GPUS[$gpu_index]}

                        # Define run name and directory based on hyperparameters
                        run_name="lr${lr}_bs${bs}_ld${ld}_hd${hd}_nl${nl}_epochs${epoch}"
                        run_dir="${SAVE_DIR_BASE}/${run_name}"
                        mkdir -p "$run_dir"
                        log_file="${run_dir}/run.log" # Use a specific log name

                        export HOME=/export/mkrishn9/


                        # Launch the pre-training job in the background
                        { python -u train_cv.py \
                            --lr "$lr" \
                            --latent_dim "$ld" \
                            --hidden_dim "$hd" \
                            --batch_size "$bs" \
                            --n_layers "$nl" \
                            --epochs "$epoch" \
                            --include_datasets "hutubs,cipic" \
                            --gpu "$GPU_ID" \
                            --save_dir "$run_dir" 2>&1 | tee "$log_file"; } &

                            job_count=$((job_count + 1))
                    done
                done
            done
        done
    done
done

# echo "All pre-training jobs launched. Waiting for the final batch to complete..."
# wait
# echo "--- ALL PRE-TRAINING JOBS FINISHED ---"

# ==============================================================================
#  FINE-TUNING on other datasets
# ==============================================================================
# echo "--- LAUNCHING FINE-TUNING EXPERIMENTS ---"

# # Reset job count
# job_count=0

# # EPOCHS=(100 200 300 400 500 600 800 1000)


# for epoch_ft in "${EPOCHS_FT[@]}"; do
#     for lr_ft in "${LEARNING_RATES_FT[@]}"; do


#         for nl in "${N_LAYERS[@]}"; do
#             for hd in "${HIDDEN_DIMS[@]}"; do
#                 for ld in "${LATENT_DIMS[@]}"; do
#                     for lr in "${LEARNING_RATES[@]}"; do
#                         for epoch in "${EPOCHS[@]}"; do
#                             if [[ "$epoch" -eq 400 && ( ("$lr" == "1e-4" ) && "$ld" -eq 64 && "$hd" -eq 64 ) ]]; then
#                                 continue
#                             fi

#                             # Define the path to the pre-trained model's directory and checkpoint
#                             run_name="lr${lr}_bs${BATCH_SIZE}_ld${ld}_hd${hd}_nl${nl}_epochs${epoch}"
#                             pretrained_run_dir="${SAVE_DIR_BASE}/${run_name}"
#                             checkpoint_path="${pretrained_run_dir}/hrtf_anthro_model_best_allfreqs.pth"


#                             if [ ! -f "$checkpoint_path" ]; then
#                                 echo "WARNING: Checkpoint not found, skipping fine-tuning for: $checkpoint_path"
#                                 continue
#                             fi


#                             while [[ $(jobs -p | wc -l) -ge $NUM_GPUS ]]; do
#                                 echo "All GPUs are busy. Waiting for a job to finish..."
#                                 wait -n
#                             done


#                             gpu_index=$((job_count % NUM_GPUS))
#                             GPU_ID=${GPUS[$gpu_index]}

#                             ft_run_name="ft_lr${lr_ft}_epochs${epoch_ft}"
#                             ft_save_dir="${pretrained_run_dir}/${ft_run_name}"
#                             mkdir -p "$ft_save_dir"

#                             # Save the log inside the new unique directory
#                             log_file_ft="${ft_save_dir}/run_ft.log"

#                             echo "Launching Fine-tuning from ${run_name} on GPU ${GPU_ID} (LR: ${lr_ft}, Epochs: ${epoch_ft})"
#                             echo "==> Saving results to: ${ft_save_dir}"

#                             export HOME=/export/mkrishn9/

#                             # Launch the fine-tuning job in the background
#                             { python -u train_ft.py \
#                                 --lr "$lr_ft" \
#                                 --epochs "$epoch_ft" \
#                                 --latent_dim "$ld" \
#                                 --hidden_dim "$hd" \
#                                 --batch_size "$BATCH_SIZE" \
#                                 --n_layers "$nl" \
#                                 --load_checkpoint "$checkpoint_path" \
#                                 --save_dir "$ft_save_dir" \
#                                 --include_datasets "hutubs,cipic,ari,scut" \
#                                 --gpu "$GPU_ID" 2>&1 | tee "$log_file_ft"; } &

#                             job_count=$((job_count + 1))
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

# --- Final Wait ---
echo "All experiments launched. Waiting for the final batch to complete..."


wait

echo "Hyperparameter sweep finished."
