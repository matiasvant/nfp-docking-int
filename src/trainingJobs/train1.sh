#!/bin/bash

cd ./src/trainingJobs
module load python-libs/3.0
python ../../reg_train.py -dropout 0.1 -learn_rate 0.001 -os 25 -bs 64 -data dock_acease_pruned -fplen 64 -wd 0.0001 -mnum 1
