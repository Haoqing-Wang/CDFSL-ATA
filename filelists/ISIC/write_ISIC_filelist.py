import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import random
import pandas as pd

cwd = os.getcwd()
data_path = join(cwd, 'ISIC2018_Task3_Training_Input')
csv_path = join(cwd, 'ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv')
savedir = './'

print('{} contains {} categories'.format('ISIC', 7))

file_list = []
label_list = []

data_info = pd.read_csv(csv_path, header=None)
folder_list = [s for s in data_info.iloc[0, :][1:]]
for i in range(len(data_info.iloc[:, 0])-1):
    img_dir = join(data_path, data_info.iloc[i+1, 0]+'.jpg')
    if isfile(img_dir):
        file_list.append(img_dir)
    label = np.asarray(data_info.iloc[i+1,1:])
    label = np.argmax(label)
    label_list.append(label)

fo = open(savedir + "novel.json", "w")
fo.write('{"label_names": [')
fo.writelines(['"%s",' % item for item in folder_list])
fo.seek(0, os.SEEK_END)
fo.seek(fo.tell() - 1, os.SEEK_SET)
fo.write('],')

fo.write('"image_names": [')
fo.writelines(['"%s",' % item for item in file_list])
fo.seek(0, os.SEEK_END)
fo.seek(fo.tell() - 1, os.SEEK_SET)
fo.write('],')

fo.write('"image_labels": [')
fo.writelines(['%d,' % item for item in label_list])
fo.seek(0, os.SEEK_END)
fo.seek(fo.tell() - 1, os.SEEK_SET)
fo.write(']}')

fo.close()
print("ISIC -OK")