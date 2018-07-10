#!/bin/bash

# cd ~/MTCNN/mtcnn-pytorch
# python gen_landmark.py
# cd ~/insightface/dataset/faces_baby_112x112/
# python reorganize_train_input_data.py   # generate train.lst

python ../../src/data/face2rec2.py . --num-thread 4
