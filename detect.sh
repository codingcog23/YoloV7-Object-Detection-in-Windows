#!/bin/bash

for file in inference/images/*.jpg
do 
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source $file --class 0

done
