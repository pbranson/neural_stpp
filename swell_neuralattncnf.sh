#!/bin/bash

#Preliminaries
cd /group/pawsey0106/pbranson/repos/neural_stpp/data
python preprocess_swell.py $1

cd /group/pawsey0106/pbranson/repos/neural_stpp/

python train_stpp.py --data swells --model attncnf --tpp neural --l2_attn --fset $1

