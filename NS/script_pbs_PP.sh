#!/bin/bash

#PBS -S /bin/bash
#PBS -q beta 
#PBS -l select=1:ncpus=8:mpiprocs=8
#PBS -l walltime=72:00:00
#PBS -N metapop_4
#PBS -j oe


cd $PBS_O_WORKDIR

module load python-3.6
source ../../../bin/activate

python3.6 -m scoop -n 8 population_priors.py > log_PP_metapop_4.txt

exit 0

