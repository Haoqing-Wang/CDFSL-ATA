import os
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F

from methods.backbone import model_dict
from data.datamgr import SetDataManager
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.protonet import ProtoNet
from methods.gnnnet import GnnNet
from methods.tpn import TPN
from options import parse_args

def guassian_kernel(source, target, kernel_mul, kernel_num, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])  # n + m
    total = torch.cat([source, target], dim=0)  # (n+m, d)
    AB = torch.mm(total, total.transpose(0, 1))  # (n+m, n+m)
    AA = (total * total).sum(dim=1, keepdim=True).expand_as(AB)  # (n+m, n+m)
    BB = (total * total).sum(dim=1).unsqueeze(0).expand_as(AB)  # (n+m, n+m)
    L2_distance = AA - 2.*AB + BB

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.detach().data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul**(kernel_num//2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def mmd(source, target, kernel_mul=2., kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss

def Max_phase(model, X_n):
    X_n = X_n.cuda()
    optimizer = optim.SGD([X_n.requires_grad_()], lr=params.max_lr)
    model.eval()
    init_features = None
    for i in range(params.T_max):
        optimizer.zero_grad()
        last_features = model.feature(X_n.reshape(-1, *X_n.size()[2:]))  # (105, 512)
        if params.method in ['RelationNet', 'TPN']:
            last_features = F.avg_pool2d(last_features, kernel_size=7).squeeze()  # [105, 512]
        if i == 0:
            init_features = last_features.clone().detach()  # (105, 512)

        _, class_loss = model.set_forward_loss(X_n)
        feature_loss = mmd(last_features, init_features)
        adv_loss = params.lamb * feature_loss - class_loss
        adv_loss.backward()
        optimizer.step()
        del last_features, class_loss, feature_loss, adv_loss
    return X_n.detach()

def train(base_loader, val_loader, model, start_epoch, stop_epoch, params):
    max_acc = 0.
    optimizer = torch.optim.Adam(model.parameters())
    print_freq = len(base_loader)//10

    for epoch in range(start_epoch, stop_epoch):
        avg_loss = 0.
        for i, (x, _) in enumerate(base_loader):  # (5, 21, 3, 224, 224)
            x_hat = Max_phase(model, x)  # (5, 21, 3, 224, 224)
            model.train()
            optimizer.zero_grad()
            _, loss = model.set_forward_loss(x_hat)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()
            if (i + 1) % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i+1, len(base_loader), avg_loss/float(i+1)))
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
    print('\ttrain with single seen domain {}'.format(params.dataset))
    print('\tval with single seen domain {}'.format(params.testset))
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
    elif params.method == 'ProtoNet':
        model = ProtoNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    elif params.method == 'GNN':
        model = GnnNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    elif params.method == 'TPN':
        model = TPN(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    else:
        print("Please specify the method!")
        assert(False)
    model.n_query = n_query

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