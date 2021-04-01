#!/bin/bash

#PBS -S /bin/bash
#PBS -q beta 
#PBS -l select=1:ncpus=24:mpiprocs=24
#PBS -l walltime=120:00:00
#PBS -N metaworld_0
#PBS -j oe


cd $PBS_O_WORKDIR

#module load python-3.6 #I'm using python 3.8 which is managed through pyenv
source ../../../bin/activate

python3.8 -m scoop -n 24 population_priors.py > log_metaworld_0.txt

exit 0

