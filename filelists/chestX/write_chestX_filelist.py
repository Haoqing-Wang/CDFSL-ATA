import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import random
import pandas as pd

cwd = os.getcwd()
data_path = join(cwd, 'images')
csv_path = join(cwd, 'Data_Entry_2017.csv')
savedir = './'
folder_list = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumothorax']
labels_maps = {'Atelectasis': 0, 'Cardiomegaly': 1, 'Effusion': 2, 'Infiltration': 3, 'Mass': 4, 'Nodule': 5, 'Pneumothorax': 6}
print('{} contains {} categories'.format('Chest_X', 7))

file_list = []
label_list = []

data_info = pd.read_csv(csv_path, skiprows=[0], header=None)
image_name_all = np.asarray(data_info.iloc[:, 0])
labels_all = np.asarray(data_info.iloc[:, 1])

for name, label in zip(image_name_all, labels_all):
    label = label.split('|')
    img_dir = join(data_path, name)
    if len(label) == 1 and (label[0] in folder_list) and isfile(img_dir):
        file_list.append(img_dir)
        label_list.append(labels_maps[label[0]])

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
print('ChestX -OK')