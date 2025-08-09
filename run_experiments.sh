#!/bin/bash

# This script runs multiple training experiments with different hyperparameters.

echo "Starting hyperparameter sweep..."

# Define the hyperparameters you want to test
LEARNING_RATES=(1e-5)
LATENT_DIMS=(64 128)
HIDDEN_DIMS=(64 128 256)
N_LAYERS=(3 4 5)
BATCH_SIZE=8
EPOCHS=1000
GPUS=(3 4 5 6 7)
job_count=0
NUM_GPUS=${#GPUS[@]}
SAVE_DIR="experiments"



# --- Launching Experiments in Batches ---
for lr in "${LEARNING_RATES[@]}"; do
  for ld in "${LATENT_DIMS[@]}"; do
    for hd in "${HIDDEN_DIMS[@]}"; do
      for nl in "${N_LAYERS[@]}"; do


        # Check if the number of running jobs equals the number of GPUs
        if (( job_count % NUM_GPUS == 0 && job_count > 0 )); then
            echo "----------------------------------------------------"
            echo "GPU batch full. Waiting for jobs to finish..."
            wait
            echo "Batch finished. Launching next set of jobs."
            echo "----------------------------------------------------"
        fi


        gpu_index=$((job_count % NUM_GPUS))
        GPU_ID=${GPUS[$gpu_index]}

        run_name="lr${lr}_bs${BATCH_SIZE}_ld${ld}_hd${hd}_nl${nl}"
        run_dir="${SAVE_DIR}/${run_name}"
        mkdir -p "$run_dir"
        log_file="${run_dir}/run.log"
        echo "Launching Run: ${run_name} on GPU ${GPU_ID}"




        { python -u train_args.py \
          --lr "$lr" \
          --latent_dim "$ld" \
          --hidden_dim "$hd" --batch_size "$BATCH_SIZE"\
          --n_layers "$nl" \
          --gpu "$GPU_ID" \
          --save_dir "$SAVE_DIR" 2>&1 | tee "$log_file"; } &




        job_count=$((job_count + 1))

      done
    done
  done
done

# --- Final Wait ---
# Wait for the last batch of jobs to complete
echo "All experiments launched. Waiting for the final batch to complete..."
wait


echo "Hyperparameter sweep finished."