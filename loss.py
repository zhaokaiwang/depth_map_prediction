import torch 
import torch.nn as nn 
import numpy as np 

class l1Loss(nn.Module):
    def __init__(self):
        super(l1Loss, self).__init__()
        pass 
    
    def forward(self, input, target, mask):
        input = input[mask]
        target = target[mask]
        count = torch.sum(mask).float()

        temp = torch.abs(input - target)
        temp = torch.sum(temp / count)
        return temp

class l2Loss(nn.Module):
    def __init__(self):
        super(l2Loss, self).__init__()

    def forward(self, input, target, mask):
        input = input[mask]
        target = target[mask]
        count = torch.sum(mask).float()
        
        temp = input - target
        temp = torch.sum(torch.pow(temp, 2))
        temp = temp / count
        return temp
        
class thresholdLoss(nn.Module):
    def __init__(self):
        super(thresholdLoss, self).__init__()
    
    def forward(self, input, target, threshold):
        top = input / target
        down = target / input
        max_one = torch.max(top, down)
        
        valid = torch.sum(max_one <= threshold)
        return valid
    
class absRelative(nn.Module):
    def __init__(self):
        super(absRelative, self).__init__()

    def forward(self, input, target):
        temp = torch.sum (torch.abs(input - target) / target)
        return temp

class squareRelative(nn.Module):
    def __init__(self):
        super(squareRelative, self).__init__()
    
    def forward(self, input, target):
        temp = torch.pow(input - target, 2)
        temp = temp / target

        return torch.sum(temp)

class rmseLinear(nn.Module):
    def __init__(self):
        super(rmseLinear, self).__init__()
    
    def forward(self, input, target):
        temp = torch.pow(input - target, 2)

        return torch.sum(temp)


class rmseLog(nn.Module):
    def __init__(self):
        super(rmseLog, self).__init__()
    
    def forward(self, input, target):
        temp = torch.log(input) - torch.log(target)
        temp = torch.sum(torch.pow(temp, 2))

        return temp

class log10Loss(nn.Module):
    def __init__(self):
        super(log10Loss, self).__init__()
    
    def forward(self, input, target):
        temp = torch.log10(input) - torch.log10(target)
        return torch.sum(torch.abs(temp))

def error_mertic(predict, target):
    device = torch.device('cuda:0')

    threshold = thresholdLoss().to(device)
    absRel = absRelative().to(device)
    sqrRel = squareRelative().to(device)
    rmsLin = rmseLinear().to(device)
    rmsLog = rmseLog().to(device)
    log10Error = log10Loss().to(device)

    thrLoss = threshold(predict, target, 1.25)
    absLoss = absRel(predict, target)
    sqrLoss = sqrRel(predict, target)
    rmsLinLoss = rmsLin(predict, target)
    rmsLogLoss = rmsLog(predict, target)
    log10 = log10Error(predict, target)

    return thrLoss, absLoss, sqrLoss, rmsLinLoss, rmsLogLoss, log10

def get_loss(loss):
    if loss not in ['l1Loss', 'l2Loss']:
        raise NotImplementedError('loss {} has not been supported'.format(loss))

    if loss is 'l1Loss':
        return l1Loss()
    elif loss is 'l2Loss':
        return l2Loss()


def main():
    pass
if __name__ == '__main__':
    pass 