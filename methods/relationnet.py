from methods import backbone
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from LRPtools import lrp_wrapper
from LRPtools import lrp_presets
from LRPtools import utils as LRPutil
import copy
import gc
import utils

class RelationNet(MetaTemplate):
  def __init__(self, model_func, n_way, n_support, loss_type='mse'):
    super(RelationNet, self).__init__(model_func, n_way, n_support, flatten=False)
    # loss function
    self.loss_type = loss_type
    if 'mse' in self.loss_type:
      self.loss_fn = nn.MSELoss()
    else:
      self.loss_fn = nn.CrossEntropyLoss()

    # metric function
    self.relation_module = RelationModule( self.feat_dim , 8, self.loss_type)
    self.method = 'RelationNet'

  def set_forward(self,x,is_feature = False):
    # get features
    z_support, z_query  = self.parse_feature(x,is_feature)
    z_support   = z_support.contiguous()
    z_proto     = z_support.view( self.n_way, self.n_support, *self.feat_dim ).mean(1)
    z_query     = z_query.contiguous().view( self.n_way* self.n_query, *self.feat_dim )

    # get relations with metric function
    z_proto_ext = z_proto.unsqueeze(0).repeat(self.n_query* self.n_way,1,1,1,1)
    z_query_ext = z_query.unsqueeze(0).repeat( self.n_way,1,1,1,1)
    z_query_ext = torch.transpose(z_query_ext,0,1)
    extend_final_feat_dim = self.feat_dim.copy()
    extend_final_feat_dim[0] *= 2
    relation_pairs = torch.cat((z_proto_ext,z_query_ext),2).view(-1, *extend_final_feat_dim)
    relations = self.relation_module(relation_pairs).view(-1, self.n_way)
    return relations

  def set_forward_loss(self, x):
    y = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
    scores = self.set_forward(x)
    if self.loss_type == 'mse':
      y_oh = utils.one_hot(y, self.n_way)
      y_oh = y_oh.cuda()
      loss = self.loss_fn(scores, y_oh)
    else:
      y = y.cuda()
      loss = self.loss_fn(scores, y)
    return scores, loss

# --- Convolution block used in the relation module ---
class RelationConvBlock(nn.Module):
  FWT = False
  def __init__(self, indim, outdim, padding = 0):
    super(RelationConvBlock, self).__init__()
    self.indim  = indim
    self.outdim = outdim
    if self.FWT:
      self.C      = backbone.Conv2d_fw(indim, outdim, 3, padding=padding)
      self.BN     = backbone.BatchNorm2d_fw(outdim, momentum=1, track_running_stats=False)
    else:
      self.C      = nn.Conv2d(indim, outdim, 3, padding = padding )
      self.BN     = nn.BatchNorm2d(outdim, momentum=1, affine=True, track_running_stats=False)
    self.relu   = nn.ReLU()
    self.pool   = nn.MaxPool2d(2)
    self.parametrized_layers = [self.C, self.BN, self.relu, self.pool]

    for layer in self.parametrized_layers:
      backbone.init_layer(layer)
    self.trunk = nn.Sequential(*self.parametrized_layers)

  def forward(self,x):
    out = self.trunk(x)
    return out

# --- Relation module adopted in RelationNet ---
class RelationModule(nn.Module):
  FWT = False
  def __init__(self,input_size,hidden_size, loss_type = 'mse'):
    super(RelationModule, self).__init__()
    self.loss_type = loss_type
    padding = 1 if ( input_size[1] <10 ) and ( input_size[2] <10 ) else 0 # when using Resnet, conv map without avgpooling is 7x7, need padding in block to do pooling
    self.layer1 = RelationConvBlock(input_size[0]*2, input_size[0], padding = padding )
    self.layer2 = RelationConvBlock(input_size[0], input_size[0], padding = padding )
    shrink_s = lambda s: int((int((s- 2 + 2*padding)/2)-2 + 2*padding)/2)

    if self.FWT:
      self.fc1 = backbone.Linear_fw( input_size[0]* shrink_s(input_size[1]) * shrink_s(input_size[2]), hidden_size )
      self.fc2 = backbone.Linear_fw( hidden_size,1)
    else:
      self.fc1 = nn.Linear( input_size[0]* shrink_s(input_size[1]) * shrink_s(input_size[2]), hidden_size )
      self.fc2 = nn.Linear( hidden_size,1)

  def forward(self,x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = out.view(out.size(0),-1)
    out = F.relu(self.fc1(out))
    if self.loss_type == 'mse':
      out = torch.sigmoid(self.fc2(out))
    elif self.loss_type == 'softmax':
      out = self.fc2(out)
    elif self.loss_type == 'LRPmse':
      out = self.fc2(out)
    return out

# --- Relationnet with LRP weighted features ---
class RelationNetLRP(RelationNet):
  def __init__(self, model_func, n_way, n_support, loss_type='LRPmse'):
    super(RelationNetLRP, self).__init__(model_func, n_way, n_support, loss_type=loss_type)
    self.preset = lrp_presets.SequentialPresetA()
    self.scale_cls = 20
    self.lrptemperature = 1
    self.method = 'RelationNetLPR'

  def get_feature_relevance(self, relation_pairs, relations):
    model = copy.deepcopy(self.relation_module)
    lrp_wrapper.add_lrp(model, preset=self.preset)
    relations_sf = torch.softmax(relations, dim=-1)
    assert not torch.isnan(relations_sf.sum())
    assert not torch.isinf(relations_sf.sum())
    relations_logits = torch.log(LRPutil.LOGIT_BETA * (relations_sf +LRPutil.EPSILON)/ (torch.tensor([1 + LRPutil.EPSILON]).cuda() - relations_sf))
    relations_logits = relations_logits.view(-1, 1)
    assert not torch.isnan(relations_logits.sum())
    assert not torch.isinf(relations_logits.sum())
    relevance_relations = model.compute_lrp(relation_pairs, target=relations_logits) # (n_way*n_querry * n_support, 2* feature_dim, f_h, f_w)
    assert not torch.isnan(relevance_relations.sum())
    assert not torch.isinf(relevance_relations.sum())
    '''normalize the prototype and the query separately '''
    relevance_prototype = relevance_relations.narrow(1, 0, self.feat_dim[0])
    relevance_query = relevance_relations.narrow(1, self.feat_dim[0], self.feat_dim[0])
    relevance_prototype = LRPutil.normalize_relevance(relevance_prototype, dim=1, temperature=1)
    relevance_query = LRPutil.normalize_relevance(relevance_query, dim=1, temperature=1)
    normalized_relevance = LRPutil.normalize_relevance(relevance_relations, dim=1, temperature=1)
    del model
    gc.collect()
    return normalized_relevance, relevance_prototype, relevance_query

  def set_forward_loss(self, x):
    y_local = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))

    scores = self.set_forward(x)
    if isinstance(scores, tuple) and len(scores) == 2:
      relations, relations_lrp = scores
      y_oh = utils.one_hot(y_local, self.n_way)
      y_oh = y_oh.cuda()
      loss1 = self.loss_fn(relations, y_oh)
      loss2 = self.loss_fn(relations_lrp, y_oh)
      if self.n_support == 5:
        loss = loss1 + loss2
      if self.n_support == 1:
        loss = loss1 + 0.5 * loss2
      return relations, loss
    elif 'mse' in self.loss_type:
      y_oh = utils.one_hot(y_local, self.n_way)
      y_oh = y_oh.cuda()
      loss = self.loss_fn(scores, y_oh)
    else:
      y_local = y_local.cuda()
      loss = self.loss_fn(scores, y_local)
    return scores, loss

  def set_forward(self, x, is_feature=False):
    z_support, z_query = self.parse_feature(x, is_feature)
    z_support = z_support.contiguous()
    z_proto = z_support.view( self.n_way, self.n_support, *self.feat_dim ).mean(1)
    z_query = z_query.contiguous().view( self.n_way* self.n_query, *self.feat_dim )

    # get relations with metric function
    z_proto_ext = z_proto.unsqueeze(0).repeat(self.n_query* self.n_way,1,1,1,1)
    z_query_ext = z_query.unsqueeze(0).repeat( self.n_way,1,1,1,1)
    z_query_ext = torch.transpose(z_query_ext,0,1)   # n_querry * n_way, n_way, *fea_dim
    extend_final_feat_dim = self.feat_dim.copy()
    extend_final_feat_dim[0] *= 2
    relation_pairs = torch.cat((z_proto_ext,z_query_ext),2).view(-1, *extend_final_feat_dim)
    relations = self.relation_module(relation_pairs).view(-1, self.n_way)

    if self.lrptraining:
      self.relation_module.eval()
      relation_pairs_lrp = relation_pairs.detach()
      relations_lrp = self.relation_module(relation_pairs_lrp).view(-1, self.n_way)
      relevance_relation_pairs, _,_ = self.get_feature_relevance(relation_pairs_lrp, relations_lrp)
      relation_pairs = relation_pairs * relevance_relation_pairs
      self.relation_module.train()
      relations_lrp = self.relation_module(relation_pairs).view(-1, self.n_way)
      relations = torch.sigmoid(relations)
      relations_lrp = torch.sigmoid(relations_lrp)
      return relations, relations_lrp

    if 'mse' in self.loss_type:
      relations = torch.sigmoid(relations)
    return relations