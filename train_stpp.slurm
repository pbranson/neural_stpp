#!/bin/bash
#SBATCH --account=pawsey0106
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --export=NONE

module load singularity
module load cuda/11.4.2
export SINGULARITYENV_CUDA_HOME=$CUDA_HOME

container=/group/pawsey0106/singularity/neural_stpp.sif

#Set these to have singularity bind data locations
export SINGULARITY_BINDPATH=/group:/group,/scratch:/scratch,/run:/run,$MYGROUP/home:$HOME

#This is needed to setup conda in the container correctly
export XDG_DATA_HOME=$MYGROUP/home/.local
export SINGULARITYENV_XDG_DATA_HOME=$XDG_DATA_HOME
export SINGULARITYENV_LD_LIBRARY_PATH=/opt/conda/lib
export SINGULARITYENV_PREPEND_PATH=/opt/conda/bin:$XDG_DATA_HOME/bin

srun --export=ALL -n 1 -N 1 -c $SLURM_CPUS_PER_TASK singularity exec --nv -C \
    $container \
    /group/pawsey0106/pbranson/repos/neural_stpp/$1 $2
