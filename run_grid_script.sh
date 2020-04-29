#!/bin/bash

for i in {0..7}
do
	qsub -l h_rt=2:00:0 -l mem=1G -wd /home/ucapsjg/workdir7 ./param_grid_script.py /home/ucapsjg/random_circuits/grids/param_grid_2d_parallel1.json $i --datadir /home/ucapsjg/random_circuits/data7
done
