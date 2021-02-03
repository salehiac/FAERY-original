#!/bin/bash

#BR-NS, also compute epsilon
#python3 -i kappa_eta_eps.py --qmat ~/misc_experiments/guidelines_log/for_open_source_code/Q_matrices_1000_gens/qm_large_ant_maze_br_ns_run_0.npz  --problem_type "br-ns" --env "large_ant" --root_dir "/home/achkan/misc_experiments/guidelines_log/for_open_source_code/large_ant/learnt/NS_log_1605/"

#BR-NS, known epsilon
python3 -i kappa_eta_eps.py --qmat ~/misc_experiments/guidelines_log/for_open_source_code/Q_matrices_1000_gens/qm_large_ant_maze_br_ns_run_0.npz  --problem_type "br-ns" --env "large_ant" --epsilon 1.49

#Archive-based, unknown epsilon
#python3 -i kappa_eta_eps.py --qmat ~/misc_experiments/guidelines_log/for_open_source_code/Q_matrices_1000_gens/qm_large_ant_maze_archive_6000_run_0.npz  --problem_type "archive_based" --env "large_ant" --skipped_qm 10

#Archive-based, unknown epsilon
#python3 -i kappa_eta_eps.py --qmat ~/misc_experiments/guidelines_log/for_open_source_code/Q_matrices_1000_gens/qm_large_ant_maze_archive_4000_run_0.npz  --problem_type "archive_based" --env "large_ant" --skipped_qm 10

#python3 -i kappa_eta_eps.py --qmat ~/misc_experiments/guidelines_log/for_open_source_code/Q_matrices_1000_gens/qm_large_ant_maze_archive_2000_run_0.npz  --problem_type "archive_based" --env "large_ant" --skipped_qm 10

