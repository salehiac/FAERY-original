#!/bin/bash

#PBS -S /bin/bash
#PBS -q beta 
#PBS -l select=2:ncpus=14:mpiprocs=14
#PBS -l walltime=01:00:00
#PBS -N metaworld_TEST
#PBS -j oe


cd $PBS_O_WORKDIR

#module load python-3.6 #I'm using python 3.8 which is managed through pyenv
source ../../../bin/activate

python3.8 -m scoop -n 28 population_priors.py > log_metaworld_TEST.txt

exit 0

