import torch
import torch.nn as nn
import numpy as np
from methods.meta_template import MetaTemplate
from methods.gnn import GNN_nl
import LRPtools.utils as LRPutil
import torch.nn.functional as F
from methods import backbone

class GnnNet(MetaTemplate):
  FWT=False
  def __init__(self, model_func,  n_way, n_support):
    super(GnnNet, self).__init__(model_func, n_way, n_support)
    # loss function
    self.loss_fn = nn.CrossEntropyLoss()

    # metric function
    self.fc = nn.Sequential(nn.Linear(self.feat_dim, 128), nn.BatchNorm1d(128, track_running_stats=False)) if not self.FWT else nn.Sequential(backbone.Linear_fw(self.feat_dim, 128), backbone.BatchNorm1d_fw(128, track_running_stats=False))
    self.gnn = GNN_nl(128 + self.n_way, 96, self.n_way)
    self.method = 'GnnNet'

  def cuda(self):
    self.feature.cuda()
    self.fc.cuda()
    self.gnn.cuda()
    return self

  def set_forward(self, x, is_feature=False):
    x = x.cuda()

    if is_feature:
      # reshape the feature tensor: n_way * n_s + 15 * f
      assert(x.size(1) == self.n_support + 15)
      z = self.fc(x.view(-1, *x.size()[2:]))
      z = z.view(self.n_way, -1, z.size(1))
    else:
      # get feature using encoder
      x = x.view(-1, *x.size()[2:])
      z = self.fc(self.feature(x))
      z = z.view(self.n_way, -1, z.size(1))

    # stack the feature for metric function: n_way * n_s + n_q * f -> n_q * [1 * n_way(n_s + 1) * f]
    z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
    assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    scores = self.forward_gnn(z_stack)
    return scores

  def forward_gnn(self, zs):
    # fix label for training the metric function   1*nw(1 + ns)*nw
    support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).unsqueeze(1)
    support_label = torch.zeros(self.n_way*self.n_support, self.n_way).scatter(1, support_label, 1).view(self.n_way, self.n_support, self.n_way)
    support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, self.n_way)], dim=1)
    support_label = support_label.view(1, -1, self.n_way).cuda()
    # gnn inp: n_q * n_way(n_s + 1) * f
    nodes = torch.cat([torch.cat([z, support_label], dim=2) for z in zs], dim=0)
    scores = self.gnn(nodes)

    # n_q * n_way(n_s + 1) * n_way -> (n_way * n_q) * n_way
    scores = scores.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0, 2).contiguous().view(-1, self.n_way)
    return scores

  def set_forward_loss(self, x):
    y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
    y_query = y_query.cuda()
    scores = self.set_forward(x)
    loss = self.loss_fn(scores, y_query)
    return scores, loss

class GnnNetLRP(GnnNet):
  FWT = False
  def __init__(self, model_func, n_way, n_support):
    super(GnnNetLRP, self).__init__(model_func, n_way, n_support)
    self.method = 'GnnNetLRP'

  def lrp_backpropagate_linear(self, relevance_output, feature_input, weight, bias=None, ignore_bias=True):
    if len(weight.size()) == 2:
      V = weight.clone().detach()
      input_ = feature_input.clone().detach()
      Z = torch.mm(input_, V.t())
      if ignore_bias:
        assert bias == None
        Z += LRPutil.EPSILON * Z.sign()
        Z.masked_fill_(Z==0, LRPutil.EPSILON)

      if not ignore_bias:
        assert bias != None
        Z += bias.clone().detach()

      S = relevance_output.clone().detach()/Z
      C = torch.mm(S, V)
      R = input_ * C
      assert not torch.isnan(R.sum())
      assert not torch.isinf(R.sum())
      return R

    elif len(weight.size()) == 3:
      bs = relevance_output.size(0)
      relevance_input = []
      for i in range(bs):
        V = weight[i].clone().detach()  # (J*N, N)
        input_ = feature_input[i].clone().detach()  # (N, num_feature)
        relevance_outputi = relevance_output[i]
        Z = torch.mm(V, input_)  # J*N, num_feature
        Z += LRPutil.EPSILON * Z.sign()  # Z.sign() returns -1 or 0 or 1
        Z.masked_fill_(Z == 0, LRPutil.EPSILON)
        S = relevance_outputi.clone().detach() / Z
        C = torch.mm(V.t(), S)
        R = input_ * C
        assert not torch.isnan(R.sum())
        assert not torch.isinf(R.sum())
        relevance_input.append(R.unsqueeze(0))
      R = torch.cat(relevance_input, 0)
      return R

  def explain_Gconv(self, relevance_output, Gconvlayer, Wi, feature_input):
    # one forward pass
    feature_input = feature_input.clone().detach()
    Wi = Wi.clone().detach()
    W_size = Wi.size()
    N = W_size[-2]
    J = W_size[-1]
    bs = W_size[0]
    W = Wi.split(1, 3)  # [tensors, each with bs N, N, 1]
    W = torch.cat(W, 1).squeeze(3)  # W is now a tensor of size (bs, J*N, N)
    output = torch.bmm(W, feature_input)  # output has size (bs, J*N, num_features)
    num_feature = output.size(-1)
    output = output.split(N, 1)
    output = torch.cat(output, 2)  # output has size (bs, N, J*num_features)
    output = output.view(-1, Gconvlayer.num_inputs)
    relevance_output = relevance_output.view(-1, Gconvlayer.num_outputs)
    relevance_output = self.lrp_backpropagate_linear(relevance_output, output, Gconvlayer.fc.weight)
    relevance_output = relevance_output.view(bs, N, J * num_feature)
    relevance_output = relevance_output.split(num_feature, -1)
    relevance_output = torch.cat(relevance_output, 1)
    relevance_feature_input = self.lrp_backpropagate_linear(relevance_output, feature_input, W)
    assert not torch.isnan(relevance_feature_input.sum())
    assert not torch.isinf(relevance_feature_input.sum())
    return relevance_feature_input

  def _get_gnnfeature_relevance(self, gnn_nodes):
    W_init = torch.eye(gnn_nodes.size(1), device=gnn_nodes.device).unsqueeze(0).repeat(gnn_nodes.size(0), 1,1).unsqueeze(3)  # (n_querry, n_way*(num_support + 1), n_way*(num_support + 1), 1)
    # the first iteration
    W1 = self.gnn._modules['layer_w{}'.format(0)](gnn_nodes, W_init)  # (n_querry, n_way*(num_support + 1), n_way*(num_support + 1), 2)
    x_new1 = F.leaky_relu(self.gnn._modules['layer_l{}'.format(0)]([W1, gnn_nodes])[1])  # (num_querry, n_way*(num_support + 1), num_outputs)
    gnn_nodes_1 = torch.cat([gnn_nodes, x_new1], 2)  # (concat more features)

    #  the second iteration
    W2 = self.gnn._modules['layer_w{}'.format(1)](gnn_nodes_1,W_init)  # (n_querry, n_way*(num_support + 1), n_way*(num_support + 1), 2)
    x_new2 = F.leaky_relu(self.gnn._modules['layer_l{}'.format(1)]([W2, gnn_nodes_1])[1])  # (num_querry, n_way*(num_support + 1), num_outputs)
    gnn_nodes_2 = torch.cat([gnn_nodes_1, x_new2], 2)  # (concat more features)
    Wl = self.gnn.w_comp_last(gnn_nodes_2, W_init)
    scores = self.gnn.layer_last([Wl, gnn_nodes_2])[1]
    scores_sf = torch.softmax(scores, dim=-1)
    gnn_logits = torch.log(LRPutil.LOGIT_BETA * (scores_sf +LRPutil.EPSILON)/ (torch.tensor([1 + LRPutil.EPSILON]).cuda() - scores_sf))
    gnn_logits_cls = gnn_logits.view(-1, self.n_way)
    relevance_gnn_nodes_2 = self.explain_Gconv(gnn_logits_cls, self.gnn.layer_last, Wl, gnn_nodes_2)
    relevance_x_new2 = relevance_gnn_nodes_2.narrow(-1, 181, 48)
    relevance_gnn_nodes_1 = self.explain_Gconv(relevance_x_new2, self.gnn._modules['layer_l{}'.format(1)], W2, gnn_nodes_1)
    relevance_x_new1 = relevance_gnn_nodes_1.narrow(-1, 133, 48)
    relevance_gnn_nodes = self.explain_Gconv(relevance_x_new1, self.gnn._modules['layer_l{}'.format(0)], W1, gnn_nodes)
    relevance_gnn_features = relevance_gnn_nodes.narrow(-1, 0, 128)
    relevance_gnn_features += relevance_gnn_nodes_1.narrow(-1, 0, 128)
    relevance_gnn_features += relevance_gnn_nodes_2.narrow(-1, 0, 128)  #[2, 30, 128]
    relevance_gnn_weight = LRPutil.normalize_relevance(relevance_gnn_features, dim=-1, temperature=1)
    return relevance_gnn_weight

  def forward_gnn(self, zs):
    # fix label for training the metric function   1*nw(1 + ns)*nw
    support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).unsqueeze(1)
    support_label = torch.zeros(self.n_way*self.n_support, self.n_way).scatter(1, support_label, 1).view(self.n_way, self.n_support, self.n_way)
    support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, self.n_way)], dim=1)
    support_label = support_label.view(1, -1, self.n_way).cuda()

    bs = len(zs)
    nodes = torch.cat([torch.cat([z, support_label], dim=2) for z in zs], dim=0)
    scores = self.gnn(nodes)
    if self.lrptraining:
      nodes_lrp = nodes.detach()
      relevance_weight = self._get_gnnfeature_relevance(nodes_lrp)
      nodes = torch.cat([torch.cat([zs[i] * relevance_weight[i], support_label], dim=2) for i in range(bs)], dim=0)
      scores_lrp = self.gnn(nodes)
      scores_lrp = scores_lrp.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0, 2).contiguous().view(-1, self.n_way)
      scores = scores.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0,2).contiguous().view(-1, self.n_way)
      return scores, scores_lrp
    scores = scores.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0, 2).contiguous().view(-1, self.n_way)
    return scores

  def set_forward_loss(self, x):
    y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
    y_query = y_query.cuda()
    scores = self.set_forward(x)
    if isinstance(scores, tuple) and len(scores) == 2:
      out, out_lrp = scores
      loss1 = self.loss_fn(out, y_query)
      loss2 = self.loss_fn(out_lrp, y_query)
      if self.n_support == 5:
        loss = loss1 + loss2
      if self.n_support == 1:
        loss = loss1 + 0.5 * loss2
    else:
      loss = self.loss_fn(scores, y_query)
    return scores, loss