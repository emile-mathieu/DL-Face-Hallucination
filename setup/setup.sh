#!/bin/bash

eval "$(conda shell.bash hook)"

conda env create -f environment.yml

set -e 
trap 'echo "Error occurred during copy"; exit 1' ERR
# i tried to expose my folder in the environment by setting permissions
# hopefully it works so you can just directly copy the folder over in the environment
# if this fails, then you need to copy the data source over yourself
parent_dir="$(dirname "$(pwd)")"
my_data_dir=/home/msai/ruijiane001/AI6301/Face-Hallucination/Deep-Learning-Face-Hallucination/celeba

cp -r my_data_dir parent_dir