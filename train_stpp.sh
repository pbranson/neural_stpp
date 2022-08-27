#!/bin/bash

#Preliminaries
cd /group/pawsey0106/pbranson/repos/neural_stpp/data
# python download_and_preprocess_earthquakes.py
#python download_and_preprocess_covid19.py &
#python download_and_preprocess_citibike.py &
#wait

cd /group/pawsey0106/pbranson/repos/neural_stpp/

data=swells
# python setup.py build_ext --inplace
# python train_stpp.py --data $data --model gmm --tpp hawkes
# python train_stpp.py --data $data --model jumpcnf --tpp neural --solve_reverse



# python train_stpp.py --data $data --model attncnf --tpp neural --l2_attn
python train_stpp.py --data $data --model jumpcnf --tpp neural --solve_reverse
