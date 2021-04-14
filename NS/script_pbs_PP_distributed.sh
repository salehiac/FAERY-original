#!/bin/bash

#PBS -S /bin/bash
#PBS -q beta 
#PBS -l select=4:ncpus=24:mpiprocs=24
#PBS -l walltime=120:00:00
#PBS -N metaworld_box_close
#PBS -j oe


cd $PBS_O_WORKDIR

#module load python-3.6 #I'm using python 3.8 which is managed through pyenv
source ../../../bin/activate

python3.8 -m scoop -n 96 population_priors.py > log_metaworld_box_close.txt
#python3.8 -m scoop -n 96 population_priors.py --resume /scratchbeta/salehia/METAWORLD_EXPERIMENTS/META_LOGS/meta-learning_GOmL1TBr1t_46545/population_prior_0 > log_metaworld_basketball.txt

exit 0

