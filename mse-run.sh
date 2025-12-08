#!/bin/sh

python train.py -s data/compiled_training_data/Case1_4DCT --iterations 30000 --model_path output/300Projections_Case1_B_2_MSE
python train.py -s data/compiled_training_data/Case2_4DCT --iterations 30000 --model_path output/300Projections_Case2_B_2_MSE
python train.py -s data/compiled_training_data/Case3_4DCT --iterations 30000 --model_path output/300Projections_Case3_B_2_MSE
python train.py -s data/compiled_training_data/Case4_4DCT --iterations 30000 --model_path output/300Projections_Case4_B_2_MSE
python train.py -s data/compiled_training_data/Case5_4DCT --iterations 30000 --model_path output/300Projections_Case5_B_2_MSE
