import os
import numpy as np
import torch
import torch.optim

from methods.backbone import model_dict
from data.datamgr import SetDataManager
from options import parse_args
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.protonet import ProtoNet
from methods.gnnnet import GnnNet
from methods.tpn import TPN

def evaluate(novel_loader, n_way=5, n_support=5):
    iter_num = len(novel_loader)
    acc_all = []
    # Model
    if params.method == 'MatchingNet':
        model = MatchingNet(model_dict[params.model], n_way=n_way, n_support=n_support).cuda()
    elif params.method == 'RelationNet':
        model = RelationNet(model_dict[params.model], n_way=n_way, n_support=n_support).cuda()
    elif params.method == 'ProtoNet':
        model = ProtoNet(model_dict[params.model], n_way=n_way, n_support=n_support).cuda()
    elif params.method == 'GNN':
        model = GnnNet(model_dict[params.model], n_way=n_way, n_support=n_support).cuda()
    elif params.method == 'TPN':
        model = TPN(model_dict[params.model], n_way=n_way, n_support=n_support).cuda()
    else:
        print("Please specify the method!")
        assert (False)
    # Update model
    checkpoint_dir = '%s/checkpoints/%s/best_model.tar' % (params.save_dir, params.name)
    state = torch.load(checkpoint_dir)['state']
    if 'FWT' in params.name:
        model_params = model.state_dict()
        pretrained_dict = {k: v for k, v in state.items() if k in model_params}
        model_params.update(pretrained_dict)
        model.load_state_dict(model_params)
    else:
        model.load_state_dict(state)
        
    # For TPN model, we compute Batch Norm statistics from the test-time support set, not the exponential moving averages.
    if params.method != 'TPN':
        model.eval()
    for ti, (x, _) in enumerate(novel_loader):  # x:(5, 20, 3, 224, 224)
        x = x.cuda()
        n_query = x.size(1) - n_support
        model.n_query = n_query
        yq = np.repeat(range(n_way), n_query)
        with torch.no_grad():
            scores = model.set_forward(x)  # (80, 5)
            _, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()  # (80, 1)
            top1_correct = np.sum(topk_ind[:,0]==yq)
            acc = top1_correct*100./(n_way*n_query)
            acc_all.append(acc)
        print('Task %d : %4.2f%%'%(ti, acc))

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('Test Acc = %4.2f +- %4.2f%%'%(acc_mean, 1.96*acc_std/np.sqrt(iter_num)))

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args()

    image_size = 224
    iter_num = 2000
    n_query = 16

    print('Loading target dataset!')
    novel_file = os.path.join(params.data_dir, params.dataset, 'novel.json')
    datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.test_n_way, n_support=params.n_shot, n_eposide=iter_num)
    novel_loader = datamgr.get_data_loader(novel_file, aug=False)

    evaluate(novel_loader, n_way=params.test_n_way, n_support=params.n_shot)
