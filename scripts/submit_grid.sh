#!/usr/bin/bash

# --- 1. Common Variables ---
PROJECT_ROOT="/home/felix_jaspersen/Repositories/bo_pr"
EXEC_DIR="${PROJECT_ROOT}/experiments"
PARTITION="oahu"
CONDA_DIR="/home/felix_jaspersen/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="PR"
LOG_DIR="${PROJECT_ROOT}/scripts/logs"

# --- 2. Define your Grid ---
# TASKS=("ackley13" "ackley53" "svm")
TASK=("ackley53")
VARIANTS=("" "_TR")
START_SEED=42
NUM_SEEDS=3
# This automatically creates an array like: (42 43 44 45 46)
SEEDS=($(seq $START_SEED $((START_SEED + NUM_SEEDS - 1))))

# --- 3. Build the Flat Configurations List ---
# We loop here on the login node to build a list of configurations.
# We join the Job Name and Seed with a colon (:) so it's safe to pass to Slurm.
CONFIGS=()
for VARIANT in "${VARIANTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CONFIGS+=("${TASK}${VARIANT}:${SEED}")
    done
done

# Calculate the maximum array index based on the number of configurations
MAX_ARRAY_ID=$((${#CONFIGS[@]} - 1))

echo "Queuing Slurm Array (0-$MAX_ARRAY_ID) for ${#CONFIGS[@]} total jobs..."

# --- 4. Submit the Job ---
sbatch <<EOF
#!/usr/bin/bash
#SBATCH -J $TASK
#SBATCH -p $PARTITION
#SBATCH -c 4
#SBATCH -t 14:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8gb
#SBATCH --array=0-$MAX_ARRAY_ID
#SBATCH --output=${LOG_DIR}/%A_%a-${TASK}.out

source $CONDA_DIR
conda activate $CONDA_ENV
cd $EXEC_DIR

# Inject the flat configurations list from the wrapper script.
# The wrapper evaluates \${CONFIGS[@]} and pastes the list here before submission.
FLAT_CONFIGS=(${CONFIGS[@]})

# Use this specific node's Array Task ID to grab its specific configuration string
CURRENT_CONFIG=\${FLAT_CONFIGS[\$SLURM_ARRAY_TASK_ID]}

# Split the string (e.g., "ackley13_TR:42") back into two variables using the colon separator
IFS=':' read JOB_NAME CURRENT_SEED <<< "\$CURRENT_CONFIG"

echo "Starting Array Task ID: \$SLURM_ARRAY_TASK_ID"
echo "Running Experiment: \$JOB_NAME | Seed: \$CURRENT_SEED"
echo "----------------------------------------"

# Execute exactly ONE job per array task. No loops needed!
python main.py \$JOB_NAME pr__ei \$CURRENT_SEED

EOF