#!/usr/bin/env bash
unzip CropDiseases.zip
rm -r dataset/dataset
rm CropDiseases.zip

python write_CropDiseases_filelist.py