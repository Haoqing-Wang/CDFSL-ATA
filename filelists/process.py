import sys
import os
from subprocess import call

if len(sys.argv) != 2:
    raise Exception('Incorrect command! e.g., python process.py DATASET [cars, cub, places, miniImagenet, plantae]')
dataset = sys.argv[1]

print('--- process ' + dataset + ' dataset ---')
if not os.path.exists(os.path.join(dataset, 'source')):
    os.makedirs(os.path.join(dataset, 'source'))
os.chdir(os.path.join(dataset, 'source'))

# download files
if dataset == 'cars':
    call('wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz', shell=True)
    call('tar -zxf cars_train.tgz', shell=True)
    call('wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz', shell=True)
    call('tar -zxf car_devkit.tgz', shell=True)
elif dataset == 'cub':
    call('wget https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz', shell=True)
    call('tar -zxf CUB_200_2011.tgz', shell=True)
elif dataset == 'places':
    call('wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar', shell=True)
    call('tar -xf places365standard_easyformat.tar', shell=True)
# Since our experiments are based on Hung-Yu Tseng's implementation, we directly use the miniImagenet and plantae datasets from his project
elif dataset == 'miniImagenet':
    call('wget http://vllab.ucmerced.edu/ym41608/projects/CrossDomainFewShot/filelists/mini_imagenet_full_size.tar.bz2', shell=True)
    call('tar -xjf mini_imagenet_full_size.tar.bz2', shell=True)
elif dataset == 'plantae':
    call('wget http://vllab.ucmerced.edu/ym41608/projects/CrossDomainFewShot/filelists/plantae.tar.gz', shell=True)
    call('tar -xzf plantae.tar.gz', shell=True)
else:
    raise Exception('No such dataset!')

# process file
os.chdir('..')
call('python write_' + dataset + '_filelist.py', shell=True)
