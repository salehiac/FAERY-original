#!/bin/bash

#PBS -S /bin/bash
#PBS -q beta 
#PBS -l select=3:ncpus=24:mpiprocs=24
#PBS -l walltime=12:00:00
#PBS -N pickplace
#PBS -j oe


cd $PBS_O_WORKDIR

source ../../../bin/activate

python3.8 -m scoop -n  72  population_priors.py  <your_arguments>

exit 0

