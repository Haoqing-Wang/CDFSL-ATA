import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from PSG import PseudoSampleGenerator

from methods.backbone import model_dict
from data.datamgr import SetDataManager
from options import parse_args

class Finetune_Linear(nn.Module):
    def __init__(self, n_way, flatten=True, leakyrelu=False):
        super(Finetune_Linear, self).__init__()
        self.feature = model_dict[params.model](flatten=flatten, leakyrelu=leakyrelu)
        self.fc = nn.Linear(self.feature.final_feat_dim, n_way)

    def forward(self, x):
        x = self.feature(x)
        x = self.fc(x)
        return x

def finetune(novel_loader, n_pseudo=75, n_way=5, n_support=5, n_query=16):
    yq = np.repeat(range(n_way), n_query)
    iter_num = len(novel_loader)
    acc_all = []

    for ti, (x, y) in enumerate(novel_loader):
        # Model
        model = Finetune_Linear(n_way).cuda()

        # Load parameters
        path = '%s/checkpoints/%s/399.tar' % (params.save_dir, params.resume_dir)
        tmp = torch.load(path)
        state = tmp['state']
        model_params = model.state_dict()
        pretrained_dict = {k: v for k, v in state.items() if k in model_params}
        model_params.update(pretrained_dict)
        model.load_state_dict(model_params)

        # Finetune
        batch_size = 4
        pseudo_size = n_way*n_support + n_pseudo
        pseudo_genrator = PseudoSampleGenerator(n_way, n_support, n_pseudo)
        loss_fn = nn.CrossEntropyLoss().cuda()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        x = x.cuda()
        xs = x[:, :n_support].reshape(-1, *x.size()[2:])  # (25, 3, 224, 224)
        pseudo_y = torch.from_numpy(np.repeat(range(n_way), n_support+n_pseudo//n_way)).cuda()  # (100)
        model.train()
        for epoch in range(params.finetune_epoch):
            pseudo_set = pseudo_genrator.generate(xs).reshape(-1, *x.size()[2:])  # (100, 3, 224, 224)
            rand_id = np.random.permutation(pseudo_size)
            for j in range(0, pseudo_size, batch_size):
                opt.zero_grad()
                selected_id = torch.from_numpy(rand_id[j: min(j+batch_size, pseudo_size)]).cuda()
                x_batch = pseudo_set[selected_id]  # (batch_size, 3, 224, 224)
                y_batch = pseudo_y[selected_id]  # (batch_size)
                scores = model(x_batch)
                loss = loss_fn(scores, y_batch)
                loss.backward()
                opt.step()

        # Test
        xq = x[:, n_support:].reshape(-1, *x.size()[2:])  # (80, 3, 224, 224)
        model.eval()
        scores = model(xq)  # (80, 5)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        
        top1_correct = np.sum(topk_ind[:, 0]==yq)
        correct_this, count_this = float(top1_correct), len(yq)
        acc = correct_this*100./count_this
        print('Task %d : %4.2f%%' % (ti, acc))
        acc_all.append(acc)

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
    n_pseudo = 75
    print('n_pseudo: ', n_pseudo)

    print('Loading target dataset!')
    novel_file = os.path.join(params.data_dir, params.dataset, 'novel.json')
    datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.test_n_way, n_support=params.n_shot, n_eposide=iter_num)
    novel_loader = datamgr.get_data_loader(novel_file, aug=False)

    finetune(novel_loader, n_pseudo=n_pseudo, n_way=params.test_n_way, n_support=params.n_shot, n_query=n_query)