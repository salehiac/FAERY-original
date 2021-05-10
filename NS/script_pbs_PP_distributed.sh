#!/bin/bash

#PBS -S /bin/bash
#PBS -q beta 
#PBS -l select=2:ncpus=24:mpiprocs=24
#PBS -l walltime=24:00:00
#PBS -N pickplacefrombasket
#PBS -j oe


cd $PBS_O_WORKDIR

#module load python-3.6 #I'm using python 3.8 which is managed through pyenv
source ../../../bin/activate

#python3.8 -m scoop -n  24 --debug population_priors.py --resume /home/salehia/META_LOGS_for_publication/META_LOGS_BK_part_1/meta-learning_EsrLIIl54H_7724/population_prior_380  > log_pickplace_from_basket.txt
#python3.8 -m scoop -n  48  population_priors.py --resume /home/salehia/META_LOGS_for_publication/META_LOGS_BK_part_1/meta-learning_EsrLIIl54H_7724/population_prior_381  > log_pickplace_from_basket.txt
python3.8 -m scoop -n  48  population_priors.py   > log_pickplace.txt

exit 0

