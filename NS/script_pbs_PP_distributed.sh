#!/bin/bash

#PBS -S /bin/bash
#PBS -q beta 
#PBS -l select=3:ncpus=24:mpiprocs=24
#PBS -l walltime=120:00:00
#PBS -N metaworld_soccer
#PBS -j oe


cd $PBS_O_WORKDIR

#module load python-3.6 #I'm using python 3.8 which is managed through pyenv
source ../../../bin/activate

python3.8 -m scoop -n 72 population_priors.py > log_metaworld_soccer.txt

exit 0

