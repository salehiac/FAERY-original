#!/bin/bash

#PBS -S /bin/bash
#PBS -q beta 
#PBS -l select=1:ncpus=24:mpiprocs=24
#PBS -l walltime=12:00:00
#PBS -N metaworld_pick_place
#PBS -j oe


cd $PBS_O_WORKDIR

#module load python-3.6 #I'm using python 3.8 which is managed through pyenv
source ../../../bin/activate

#python3.8 -m scoop -n 72 population_priors.py > log_metaworld_pick_place.txt
python3.8 -m scoop -n 24 population_priors.py --resume /scratchbeta/salehia/METAWORLD_EXPERIMENTS/META_LOGS/meta-learning_1M48yR21f2_3064/population_prior_3 > log_metaworld_pick_place.txt

exit 0

