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
        temp = torch.sum(temp) / count
        return temp


class smoothL1Loss(nn.Module):
    def __init__(self):
        super(smoothL1Loss, self).__init__()
        self.loss = torch.nn.SmoothL1Loss(reduction='elementwise_mean')
    
    def forward(self, input, target, mask):
        input = input[mask]
        target = target[mask]

        return self.loss(input, target)

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

class berhuLoss(nn.Module):
    def __init__(self):
        super(berhuLoss, self).__init__()

    def forward(self, input, target, mask):
        input = input[mask]
        target = target[mask]
        count = torch.sum(mask).float()
        
        temp = torch.abs(input - target)
        max_one = torch.max(temp) / 5
        max_value = float(max_one.detach().cpu().numpy())

        less_part = temp <= max_one
        bigger_part = temp > max_one

        loss = torch.sum(temp[less_part])
        bigger_part = temp[bigger_part]

        bigger_part_loss = torch.sum(bigger_part * bigger_part + max_value * max_value)
        bigger_part_loss = bigger_part / 2 / max_value
        loss = torch.sum(bigger_part_loss) + loss

        return loss / count

class huberLoss(nn.Module):
    def __init__(self):
        super(huberLoss, self).__init__()
    
    def forward(self, input, target, mask):
        input = input[mask]
        target = target[mask]

        temp = torch.abs(input - target)
        max_one = torch.max(temp) / 10
        
        max_value = float(max_one.detach().cpu().numpy())
        less_part = temp <= max_one
        bigger_part = temp > max_one

        less_count = torch.sum(less_part).float()
        bigger_count = torch.sum(bigger_part).float()

        loss = torch.sum(temp[bigger_part]) / bigger_count

        less_part = temp[less_part]
        less_part_loss = less_part * less_part + max_value * max_value
        less_part_loss = less_part_loss / 2 / max_value
        loss = torch.sum(less_part_loss) / less_count + loss 
        
        return loss


class l1l2Loss(nn.Module):
    def __init__(self):
        super(l1l2Loss, self).__init__()
        self.l1 = l1Loss()
        self.l2 = l2Loss()
        self.lamda = torch.tensor([0.5]).float().cuda()

    def forward(self, input, target, mask):
        return self.l1(input, target, mask) + self.lamda * self.l2(input, target, mask)
        
        
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
    if loss not in ['l1Loss', 'l2Loss', 'berhuLoss', 'l1l2Loss', 'huberLoss', 'smoothL1']:
        raise NotImplementedError('loss {} has not been supported'.format(loss))

    if loss is 'l1Loss':
        return l1Loss()
    elif loss is 'l2Loss':
        return l2Loss()
    elif loss is 'berhuLoss':
        return berhuLoss()
    elif loss is 'l1l2Loss':
        return l1l2Loss()
    elif loss is 'huberLoss':
        return huberLoss()
    elif loss is 'smoothL1':
        return smoothL1Loss()


def main():
    
    pass
if __name__ == '__main__':
    pass 