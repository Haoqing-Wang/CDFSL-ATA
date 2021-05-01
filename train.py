import numpy as np
import torch
import torch.optim
import os

from methods.backbone import model_dict
from data.datamgr import SetDataManager
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet, RelationNetLRP
from methods.protonet import ProtoNet
from methods.gnnnet import GnnNet, GnnNetLRP
from methods.tpn import TPN
from options import parse_args

def train(base_loader, val_loader, model, start_epoch, stop_epoch, params):
    optimizer = torch.optim.Adam(model.parameters())
    max_acc = 0.
    total_it = 0
    for epoch in range(start_epoch, stop_epoch):
        model.train()
        total_it = model.train_loop(epoch, base_loader, optimizer, total_it)
        model.eval()
        with torch.no_grad():
            acc = model.test_loop(val_loader)
        if acc > max_acc:
            print("Best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
        else:
            print("GG! Best accuracy {:f}".format(max_acc))

        if ((epoch+1) % params.save_freq == 0) or (epoch == stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
    return model

# --- main function ---
if __name__=='__main__':
    # set numpy random seed
    np.random.seed(10)

    # parser argument
    params = parse_args()
    print('--- Training ---\n')
    print(params)

    # output and tensorboard dir
    params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # dataloader
    print('\n--- Prepare dataloader ---')
    print('\ttrain with seen domain {}'.format(params.dataset))
    print('\tval with seen domain {}'.format(params.testset))
    base_file = os.path.join(params.data_dir, params.dataset, 'base.json')
    val_file = os.path.join(params.data_dir, params.testset, 'val.json')

    # model
    image_size = 224
    n_query = max(1, int(16*params.test_n_way/params.train_n_way))
    base_datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.train_n_way, n_support=params.n_shot)
    base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
    val_datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.test_n_way, n_support=params.n_shot)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)

    if params.method == 'MatchingNet':
        model = MatchingNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    elif params.method == 'RelationNet':
        model = RelationNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    elif params.method == 'RelationNetLRP':
        model = RelationNetLRP(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    elif params.method == 'ProtoNet':
        model = ProtoNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    elif params.method == 'GNN':
        model = GnnNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    elif params.method == 'GNNLRP':
        model = GnnNetLRP(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    elif params.method == 'TPN':
        model = TPN(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    else:
        print("Please specify the method!")
        assert(False)

    # load model
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.resume_epoch > 0:
        resume_file = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(params.resume_epoch))
        tmp = torch.load(resume_file)
        start_epoch = tmp['epoch']+1
        model.load_state_dict(tmp['state'])
        print('\tResume the training weight at {} epoch.'.format(start_epoch))
    else:
        path = '%s/checkpoints/%s/399.tar' % (params.save_dir, params.resume_dir)
        state = torch.load(path)['state']
        model_params = model.state_dict()
        pretrained_dict = {k: v for k, v in state.items() if k in model_params}
        print(pretrained_dict.keys())
        model_params.update(pretrained_dict)
        model.load_state_dict(model_params)

    # training
    print('\n--- start the training ---')
    model = train(base_loader, val_loader, model, start_epoch, stop_epoch, params)