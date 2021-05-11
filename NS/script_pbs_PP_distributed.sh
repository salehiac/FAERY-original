#!/bin/bash

#PBS -S /bin/bash
#PBS -q beta 
#PBS -l select=3:ncpus=24:mpiprocs=24
#PBS -l walltime=12:00:00
#PBS -N pickplace
#PBS -j oe


cd $PBS_O_WORKDIR

#module load python-3.6 #I'm using python 3.8 which is managed through pyenv
source ../../../bin/activate

#when using --resume, make sure that the file is on scratchbeta or scrathalpha, NOT your home (otherwise it often can't find it)
python3.8 -m scoop -n  72 population_priors.py --resume /scratchbeta/salehia/population_prior_381 > log_pickplace.txt
#python3.8 -m scoop -n  72  population_priors.py   > log_pickplace.txt

exit 0

