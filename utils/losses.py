import torch
import torch.nn as nn
import torch.nn.functional as F 

class LabelSmoothing(nn.Module):
    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        # self.criterion = nn.KLDivLoss(reduction='batchmean')

        self.confidence = 1 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target, warm=False):

        assert x.size(1) == self.size, "label size is incorrect"
        if warm:
            return F.cross_entropy(x, target)
        true_dist = x.detach().clone()
        true_dist.fill_(self.smoothing / (self.size-1))

        
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        self.true_dist = true_dist
        loss = - torch.sum( true_dist * F.log_softmax(x, dim=1), dim=1)

        return loss.mean()


class ConfidentPenalty(nn.Module):
    def __init__(self, beta, threshold=1.5):
        super(ConfidentPenalty, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.threshold = threshold
        self.beta = beta

    def forward(self, x, target,  warm=False):
        ce = self.criterion(x, target)
        if warm:
            return ce
        u = x.detach().clone()
        u.fill_(1.0/x.size(1))
        px = F.softmax(x, dim=1)
       
        #px = torch.clamp(px, 1e-8, 1)
        Hq = - torch.sum(px * px.log(), dim=1)
        #penalty = F.kl_div(u.log(), px, reduction='batchmean')
        #print(Hq)
        # only when the h(p) is larger than threshold
        
        loss = ce - self.beta * Hq.mean()
        return loss 

class NaRCriterion(nn.Module):
    def __init__(self, beta=0.1, lam=0.5, cb_mode='kl', nb_classes=10, epislon=0.1):
        super(NaRCriterion, self).__init__()
        self.beta = beta
        self.lam  = lam
        
        self.baisc_criterion = nn.CrossEntropyLoss()
        self.lsr_criterion = LabelSmoothing(nb_classes, epislon)
        self.reg_criterion = self._kl_prob_loss if cb_mode == 'kl' else self._soft_ce_loss

    def _kl_prob_loss(self, x, y):
        x = torch.clamp(x, 1e-8,1)
        y = torch.clamp(y, 1e-8,1)
        z = x* torch.log(x/y) + y*torch.log(y/x)
        return torch.mean(torch.sum(z,dim=1),dim=0)
    
    def _soft_ce_loss(self, x, y, T=1.0):
        #x = F.softmax(x/T, dim=1)
        #y = F.softmax(y.detach()/T, dim=1)
        x = torch.clamp(x, 1e-8, 1)
        y = torch.clamp(y, 1e-8, 1)
        z = - y * torch.log(x)
        return torch.mean(torch.sum(z, dim=1), dim=0)
    
    def forward(self, preds, target, warm=False):
        loss = 0
        # main target cross entropy loss
        loss += F.cross_entropy(preds[0], target.squeeze())
        # auxiliary network (LSR)
        loss += self.lsr_criterion(preds[-1], target)
        
        if warm:
            return loss
        else:
            # construct uncertainty label
            x = F.softmax(preds[0], dim=1)
            y = F.softmax(preds[1], dim=1)
            z = (self.beta * x) + (1-self.beta)*y
            z = z.detach()
            reg_loss = self.reg_criterion(x, z)

            return (self.lam) * loss + (1 - self.lam) * reg_loss

class BootStrapping(nn.Module):
    def __init__(self, beta, mode='soft'):
        super(BootStrapping, self).__init__()
        self.beta = beta
        self._loss = self.__soft_loss if mode == 'soft' else self.__hard_loss
        self.criterion = nn.CrossEntropyLoss()

    def __soft_loss(self, logits, targets):
        eps = 1e-8
        prob = F.softmax(logits, dim=1)
        basic_loss = F.cross_entropy(logits, targets)
        soft_loss  = - torch.sum(prob * F.log_softmax(logits, dim=1), dim=1)
        loss = self.beta * basic_loss + (1 - self.beta) * soft_loss
        return loss.mean()

    def __hard_loss(self, logits, targets):
        _, idx = logits.max(1)
        basic_loss = F.cross_entropy(logits, targets)
        hard_loss  = F.cross_entropy(logits, idx.squeeze())
        loss = self.beta * basic_loss + (1 - self.beta) * hard_loss
        return loss

    def forward(self, logits, targets, warm=False):
        if warm:
            return F.cross_entropy(logits, targets)
        return self._loss(logits, targets)



def test():
    import torch

    x=torch.rand(3, 5)
    y = torch.LongTensor(3).random_(5)

    criterion = LabelSmoothing(5, 0.1)
    loss = criterion(x, y)
    #print(y)
    print(criterion.true_dist)
    print(loss)

def test2():
    import torch
    x = torch.rand(3,4)*10
    #x.fill_(0.25)
    y = torch.LongTensor(3).random_(4)
    entropy = - torch.sum(F.softmax(x, dim=1) * F.log_softmax(x, dim=1), dim=1)
    #print(entropy)
    # criterion = ConfidentPenalty(0.2)
    criterion = TransferLoss(1, 1.0, 'soft')
    x2 = torch.rand(3,4)
    preds = [x, x2]
    print(criterion(preds,y))

if __name__=='__main__':
    test2()