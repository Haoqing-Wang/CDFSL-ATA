#!/usr/bin/env bash
wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip
wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip

unzip ISIC2018_Task3_Training_GroundTruth.zip
unzip ISIC2018_Task3_Training_Input.zip

rm ISIC2018_Task3_Training_GroundTruth.zip
rm ISIC2018_Task3_Training_Input.zip

python write_ISIC_filelist.py