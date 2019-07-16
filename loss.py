import torch 
import torch.nn as nn 
import numpy as np 
import copy
import torch.nn.functional as F
import torchvision


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


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, input):
        return (input - self.mean) / self.std

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
    
class normalGanLoss(nn.Module):
    def __init__(self, mode):
        super(normalGanLoss, self).__init__()
        if mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss
        else:
            raise NotImplementedError("gan mode %s not implemented" % mode)
    
    def forward(self, predict, label):
        target_tensor = torch.tensor(label).cuda()
        target_tensor = target_tensor.expand_as(predict)
        loss = self.loss(predict, target_tensor)
        return loss
        
def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

class SytleLoss(nn.Module):
    def __init__(self, target):
        super(SytleLoss, self).__init__()
        self.target = gram_matrix(target).detach()
    
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

def perceptual_loss(cnn, noramlization_mean, noramlization_std, 
                    style_image, content_image, 
                    content_layers,
                    style_layers):
    """Get the perceptual loss of images,
    Parameters:
    cnn: the convolutional neural networks to extract features,
    normalization_mean: the input image means in vgg networks
    normalization_std: the input iamge std in vgg networks
    style_image: the style images to transfer 
    content_image: the content images to keep
    content_layers: the layers to extract features 
    style_layers: the layers to extract featurres
    """
    
    cnn = copy.deepcopy(cnn)

    noramlization = Normalization(noramlization_mean, noramlization_std)
    
    content_losses = []
    style_losses = []

    model = nn.Sequential(noramlization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        
        model.add_module(name, layer)
    
        if name in content_layers:
            target = model(content_image).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)
        
        if name in style_layers:
            target = model(style_image).detach()
            style_loss = SytleLoss(target)
            model.add_module('style_loss_{}'.format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], SytleLoss):
            break
    model = model[:(i + 1)]

    return model, content_losses, style_losses

def get_perceptual_loss(cnn, content_images, style_images):
    device = torch.device('cuda:0')
    content_layers_default = ['relu_1']
    style_layers_default = []
    #style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    return perceptual_loss(cnn, cnn_normalization_mean, cnn_normalization_std,
                           style_images, content_images, content_layers_default,
                           style_layers_default)

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
    if loss not in ['l1Loss', 'l2Loss', 'berhuLoss', 'l1l2Loss',
                    'huberLoss', 'smoothL1', 'perceptualLoss', 'normalGanLoss', 'lsganLoss']:
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
    elif loss is 'perceptualLoss':
        return get_perceptual_loss()
    elif loss is 'normalGanLoss':
        return normalGanLoss('vanilla')
    elif loss is 'lsganLoss':
        return normalGanLoss('lsgan')


def main():
    
    pass
if __name__ == '__main__':
    pass 