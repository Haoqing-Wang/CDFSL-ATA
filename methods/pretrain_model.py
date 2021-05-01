import torch.nn as nn

class Pretrain(nn.Module):
  def __init__(self, model_func, num_class):
    super(Pretrain, self).__init__()
    self.feature = model_func()
    self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
    self.classifier.bias.data.fill_(0)
    self.loss_fn = nn.CrossEntropyLoss()
    self.num_class = num_class

  def forward(self, x):
    x = x.cuda()
    out = self.feature(x)
    scores = self.classifier(out)
    return scores

  def forward_loss(self, x, y):
    scores = self.forward(x)
    y = y.cuda()
    return self.loss_fn(scores, y)

  def train_loop(self, epoch, train_loader, optimizer):
    print_freq = len(train_loader)//10
    avg_loss = 0

    for i, (x, y) in enumerate(train_loader):
      optimizer.zero_grad()
      loss = self.forward_loss(x, y)
      loss.backward()
      optimizer.step()
      avg_loss = avg_loss+loss.item()

      if (i+1) % print_freq==0:
        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i+1, len(train_loader), avg_loss/float(i+1)))