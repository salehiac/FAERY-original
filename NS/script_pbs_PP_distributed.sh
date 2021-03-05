#!/bin/bash

#PBS -S /bin/bash
#PBS -q beta 
#PBS -l select=4:ncpus=24:mpiprocs=24
#PBS -l walltime=24:00:00
#PBS -N metapop_dist
#PBS -j oe


cd $PBS_O_WORKDIR

module load python-3.6
source ../../../bin/activate

python3.6 -m scoop -n 96 population_priors.py > log_PP_metapop_dist.txt

exit 0

