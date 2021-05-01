#!/usr/bin/env bash
mkdir ./Chest_X
mkdir ./images
unzip Chest_X.zip -d Chest_X

mv ./Chest_X/images_001/images/* ./images/
mv ./Chest_X/images_002/images/* ./images/
mv ./Chest_X/images_003/images/* ./images/
mv ./Chest_X/images_004/images/* ./images/
mv ./Chest_X/images_005/images/* ./images/
mv ./Chest_X/images_006/images/* ./images/
mv ./Chest_X/images_007/images/* ./images/
mv ./Chest_X/images_008/images/* ./images/
mv ./Chest_X/images_009/images/* ./images/
mv ./Chest_X/images_010/images/* ./images/
mv ./Chest_X/images_011/images/* ./images/
mv ./Chest_X/images_012/images/* ./images/
mv ./Chest_X/Data_Entry_2017.csv ./

rm -r Chest_X
rm Chest_X.zip

python write_chestX_filelist.py