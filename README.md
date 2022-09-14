# Overview

This is the official implementation of the method proposed in 

```
@article{salehi2022few,
  title={Few-shot Quality-Diversity Optimization},
  author={Salehi, Achkan and Coninx, Alexandre and Doncieux, Stephane},
  journal={IEEE Robotics and Automation Letters},
  year={2022},
  publisher={IEEE}
}.
```

https://arxiv.org/pdf/2109.06826.pdf

# Requirements

- Python 3.6+
- Metaworld v2 benchmark (https://github.com/rlworkgroup/metaworld)
- In case you want to reproduce the experiments based on random maze distributions:

    - LibFastSim (https://github.com/jbmouret/libfastsim)
    - PyFastSim (https://github.com/alexendy/pyfastsim)
    - fastSimGym (https://github.com/alexendy/fastsim\_gym)

Other requirements can be found in `requirements.txt`.

# Usage

The results of the paper can be reproduced via

```
$ cd NS
$ python -m scoop -n <num_procs>  population_priors.py  <args>
```

For the maze experiments, you will need to generate random mazes using the script `FAERY-original/environments/maze_generator/maze_generator.py`.

