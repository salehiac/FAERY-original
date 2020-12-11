#!/bin/bash

for i in {1..30};
do
	echo "********************************************************************* experiment ${i} ********************"
	python3 -im scoop NS.py --config ../config/hardmaze_experiment_1.yaml
done
