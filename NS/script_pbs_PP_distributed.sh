#!/bin/bash

#PBS -S /bin/bash
#PBS -q beta 
#PBS -l select=4:ncpus=24:mpiprocs=24
#PBS -l walltime=120:00:00
#PBS -N ml10
#PBS -j oe


cd $PBS_O_WORKDIR

#module load python-3.6 #I'm using python 3.8 which is managed through pyenv
source ../../../bin/activate

python3.8 -m scoop -n  96 population_priors.py > log_ml10.txt
#python3.8 -m scoop -n  72 population_priors.py --resume /scratchbeta/salehia/METAWORLD_EXPERIMENTS/META_LOGS_ML10/meta-learning_mA9vhNw8Uz_30530/population_prior_9 > log_ml10.txt

exit 0

