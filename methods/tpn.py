import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from methods.meta_template import MetaTemplate
from methods.backbone import Linear_fw, Conv2d_fw, BatchNorm2d_fw

class RelationNetwork(nn.Module):
  FWT = False
  def __init__(self):
    super(RelationNetwork, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(512, 512, kernel_size=3, padding=1) if not self.FWT else Conv2d_fw(512, 512, kernel_size=3, padding=1),
      nn.BatchNorm2d(512) if not self.FWT else BatchNorm2d_fw(512),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, padding=1))
    self.layer2 = nn.Sequential(
      nn.Conv2d(512, 1, kernel_size=3, padding=1) if not self.FWT else Conv2d_fw(512, 1, kernel_size=3, padding=1),
      nn.BatchNorm2d(1) if not self.FWT else BatchNorm2d_fw(1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, padding=1))

    self.fc3 = nn.Linear(9, 18) if not self.FWT else Linear_fw(9, 18)
    self.fc4 = nn.Linear(18, 1) if not self.FWT else Linear_fw(18, 1)

  def forward(self, x):  # [105, 512, 7, 7]
    out = self.layer1(x)  # [105, 512, 4, 4]
    out = self.layer2(out)  # [105, 1, 3, 3]
    out = out.reshape(out.size(0), -1)  # [105, 9]
    out = self.fc4(F.relu(self.fc3(out)))  # [105, 1]
    return out

class TPN(MetaTemplate):
  def __init__(self, model_func, n_way, n_support):
    super(TPN, self).__init__(model_func, n_way, n_support, flatten=False)
    self.loss_fn = nn.CrossEntropyLoss()
    self.relation = RelationNetwork()
    self.alpha = 0.99
    self.method = 'TPN'

  def cuda(self):
    self.feature.cuda()
    self.relation.cuda()
    return self

  def set_forward(self, x, is_feature=False):
    x = x.cuda()  # [5, 21, 3, 224, 224]
    x = x.reshape(-1, *x.size()[2:])  # [105, 3, 224, 224]
    z = self.feature(x)  # [105, 512, 7, 7]
    z = z.reshape(self.n_way, -1, *z.size()[1:])  # [5, 21, 512, 7, 7]
    z_support = z[:, :self.n_support].reshape(-1, *z.size()[2:])  # [25, 512, 7, 7]
    z_query = z[:, self.n_support:].reshape(-1, *z.size()[2:])  # [80, 512, 7, 7]
    ys = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))  # [25]
    ys_onehot = torch.zeros(self.n_way*self.n_support, self.n_way).scatter_(1, ys.unsqueeze(1), 1).cuda()  # [25, 5]
    scores = self.forward_tpn(z_support, ys_onehot, z_query)  # [80, 5]
    return scores

  def forward_tpn(self, support, ys, query):
    eps = np.finfo(float).eps

    # Step1: Embedding
    inp = torch.cat((support, query), 0)  # [105, 512, 7, 7]
    emb_all = F.avg_pool2d(inp, kernel_size=7).squeeze()  # [105, 512]
    N, d = emb_all.size()  # 105, 512

    # Step2: Graph Construction
    self.sigma = self.relation(inp)  # [105, 1]
    emb_all = emb_all/(self.sigma+eps)  # [105, 512]
    emb1 = torch.unsqueeze(emb_all, 1)  # [105, 1, 512]
    emb2 = torch.unsqueeze(emb_all, 0)  # [1, 105, 512]
    W = ((emb1-emb2)**2).mean(2)  # [105, 105]
    W = torch.exp(-W/2.)  # [105, 105]

    topk, indices = torch.topk(W, 20)
    mask = torch.zeros_like(W)
    mask = mask.scatter(1, indices, 1)
    mask = ((mask+torch.t(mask))>0).type(torch.float32)  # union, kNN graph
    W = W * mask  # [105, 105]

    D = W.sum(0)  # [105]
    D_sqrt_inv = torch.sqrt(1./(D+eps))  # [105]
    D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N)  # [105, 105]
    D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(N, 1)  # [105, 105]
    S = D1 * W * D2  # [105, 105]

    # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
    # yu = torch.zeros(self.n_way*self.n_query, self.n_way).cuda()  # [80, 5]
    yu = (torch.ones(self.n_way*self.n_query, self.n_way)/self.n_way).cuda()  # [80, 5]
    y = torch.cat((ys, yu), 0)  # [105, 5]
    scores = torch.matmul(torch.inverse(torch.eye(N).cuda() - self.alpha * S + eps), y)  # [105, 5]
    return scores[self.n_way*self.n_support:, :]

  def set_forward_loss(self, x):
    y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).cuda()  # [80]
    scores = self.set_forward(x)  # [80, 5]
    loss = self.loss_fn(scores, y_query)
    return scores, loss