#!/usr/bin/bash

# Common variables
PROJECT_ROOT="/home/felix_jaspersen/Repositories/bo_pr"
EXEC_DIR="${PROJECT_ROOT}/experiments"
PARTITION="oahu"
# PARTITION="hawaii"
CONDA_DIR="/home/felix_jaspersen/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="PR"
LOG_DIR="${PROJECT_ROOT}/scripts/logs"

# 1. Define the parameters for your grid
TASKS=("ackley13" "ackley53" "svm")
VARIANTS=("" "_TR") # Empty string for base, _TR for variant
SEEDS=(42 43 44)

# 2. Loop through all combinations
for TASK in "${TASKS[@]}"; do
    for VARIANT in "${VARIANTS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            
            # Combine task and variant (e.g., ackley13 + _TR = ackley13_TR)
            JOB_NAME="${TASK}${VARIANT}"
            
            echo "Queuing experiment: $JOB_NAME | Seed: $SEED"

            # 3. Submit the job using a heredoc
            sbatch <<EOF
#!/usr/bin/bash
#SBATCH -J $JOB_NAME
#SBATCH -p $PARTITION
#SBATCH -c 4
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8gb
#SBATCH --output=${LOG_DIR}/%J-${JOB_NAME}-seed${SEED}.out

source $CONDA_DIR
conda activate $CONDA_ENV

cd $EXEC_DIR
python main.py $JOB_NAME pr__ei $SEED

EOF
            
            # Optional: Add a brief sleep to avoid overwhelming the scheduler
            sleep 0.2
            
        done
    done
done

echo "All jobs submitted!"
