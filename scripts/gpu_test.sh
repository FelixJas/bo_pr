#!/usr/bin/bash

# Common variables
PROJECT_ROOT="/home/felix_jaspersen/Repositories/bo_pr"
EXEC_DIR="${PROJECT_ROOT}/experiments"
PARTITION="oahu"
# PARTITION="hawaii"
CONDA_DIR="/home/felix_jaspersen/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="PR"
LOG_DIR="${PROJECT_ROOT}/scripts/logs"

JOB_NAME="GPU_Test"
SEED=0
echo "Queuing batch size experiment: $JOB_NAME"

sbatch <<EOF
#!/usr/bin/bash
#SBATCH -J $JOB_NAME
#SBATCH -p $PARTITION
#SBATCH -c 4
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8gb
#SBATCH --output=${LOG_DIR}/%J-${JOB_NAME}.out

source $CONDA_DIR
conda activate $CONDA_ENV

cd $EXEC_DIR
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())" nvidia-smi

EOF