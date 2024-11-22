#!/bin/bash
#SBATCH --job-name=TestCNN
#SBATCH --account=ec16
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

# load the Anaconda3
module load Miniconda3/22.11.1-1

# Set the ${PS1} (needed in the source of the Anaconda environment)
export PS1=\$

# Source the conda environment setup
# The variable ${EBROOTANACONDA3} or ${EBROOTMINICONDA3}
# So use one of the following lines
# comes with the module load command
# source ${EBROOTANACONDA3}/etc/profile.d/conda.sh
source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh

# Deactivate any spill-over environment from the login node
conda deactivate &>/dev/null

# Activate the environment by using the full path (not name)
# to the environment. The full path is listed if you do
# conda info --envs at the command prompt.
conda activate /fp/homes01/u01/ec-evengar/.conda/envs/pthree-dev

echo "Environment loaded successfully, running script." > templog.txt

python img_pipeline-slurm.py 256

echo "Pipeline finished" >> templog.txt