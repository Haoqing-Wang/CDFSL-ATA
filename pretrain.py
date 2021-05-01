import os
import numpy as np
import torch
import torch.optim

from methods.backbone import model_dict
from data.datamgr import SimpleDataManager
from methods.pretrain_model import Pretrain
from options import parse_args

def train(base_loader, model, start_epoch, stop_epoch, params):
    optimizer = torch.optim.Adam(model.parameters())
    # start
    model.train()
    for epoch in range(start_epoch, stop_epoch):
        model.train_loop(epoch, base_loader, optimizer)

        if ((epoch+1)%params.save_freq==0) or (epoch==stop_epoch-1):
            print('Save model...')
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
        else:
            print('-'*50)
    return model

# --- main function ---
if __name__=='__main__':
    # set numpy random seed
    np.random.seed(10)

    # parser argument
    params = parse_args()
    print('--- Pre-training ---\n')
    print(params)

    # output and tensorboard dir
    params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # dataloader
    print('\n--- prepare dataloader ---')
    print('\tPretrain with seen domain {}'.format(params.dataset))
    base_file = os.path.join(params.data_dir, params.dataset, 'base.json')

    # model
    print('\n--- build model ---')
    image_size = 224
    base_datamgr = SimpleDataManager(image_size, batch_size=16)
    base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

    model = Pretrain(model_dict[params.model], params.num_classes).cuda()

    # load model
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.resume_epoch>0:
        resume_file = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(params.resume_epoch))
        tmp = torch.load(resume_file)
        start_epoch = tmp['epoch']+1
        model.load_state_dict(tmp['state'])
        print('\tResume the training weight at {} epoch.'.format(start_epoch))

    # training
    print('\n--- start the training ---')
    model = train(base_loader, model, start_epoch, stop_epoch, params)