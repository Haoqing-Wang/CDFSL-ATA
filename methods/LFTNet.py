import torch
import torch.nn as nn
from methods import backbone
from methods.backbone import model_dict
from methods.matchingnet import MatchingNet
from methods.protonet import ProtoNet
from methods import relationnet
from methods import gnn
from methods import gnnnet
from methods import tpn

class LFTNet(nn.Module):
  def __init__(self, params):
    super(LFTNet, self).__init__()
    backbone.FeatureWiseTransformation2d_fw.feature_augment = True
    backbone.ConvBlock.FWT = True
    backbone.SimpleBlock.FWT = True
    backbone.ResNet.FWT = True

    if params.method == 'ProtoNet':
      model = ProtoNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot)
    elif params.method == 'MatchingNet':
      backbone.LSTMCell.FWT = True
      model = MatchingNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot)
    elif params.method == 'RelationNet':
      relationnet.RelationConvBlock.FWT = True
      relationnet.RelationModule.FWT = True
      model = relationnet.RelationNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot)
    elif params.method == 'GNN':
      gnnnet.GnnNet.FWT=True
      gnn.Gconv.FWT=True
      gnn.Wcompute.FWT=True
      model = gnnnet.GnnNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot)
    elif params.method == 'TPN':
        tpn.RelationNetwork.FWT = True
        model = tpn.TPN(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    else:
      raise ValueError('Unknown method')
    self.model = model
    print('\ttrain with {} framework'.format(params.method))

    # optimizer
    model_params = self.split_model_parameters()
    self.model_optim = torch.optim.Adam(model_params)

    # total epochs
    self.total_epoch = params.stop_epoch

  def split_model_parameters(self):
    model_params = []
    for n, p in self.model.named_parameters():
      n = n.split('.')
      if n[-1] == 'gamma' or n[-1] == 'beta':
        continue
      model_params.append(p)
    return model_params

  def train_loop(self, epoch, base_loader, total_it):
    self.model.train()
    for weight in self.model.parameters():
      weight.fast = None

    # trainin loop
    print_freq = len(base_loader)//10
    avg_model_loss = 0.
    for i, (x, _) in enumerate(base_loader):
      self.model.n_query = x.size(1) - self.model.n_support
      _, model_loss = self.model.set_forward_loss(x)

      # optimize
      self.model_optim.zero_grad()
      model_loss.backward()
      self.model_optim.step()

      # loss
      avg_model_loss += model_loss.item()
      if (i+1)%print_freq==0:
        print('Epoch {:d}/{:d} | Batch {:d}/{:d} | loss {:f}'.format(epoch+1, self.total_epoch, i+1, len(base_loader), avg_model_loss/float(i+1)))
      total_it += 1
    return total_it

  def test_loop(self, test_loader, record=None):
    self.model.eval()
    for weight in self.model.parameters():
      weight.fast = None
    return self.model.test_loop(test_loader, record)

  def cuda(self):
    self.model.cuda()
    return self