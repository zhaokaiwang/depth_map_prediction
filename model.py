import numpy as np 
import torch 
from densenet import densenet_cat, densenet_fc, densenet_fcn
from resnet import resnet_fc


def get_optim(model, op, lr):
    if op is 'momentum':
        optim = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9)
        return optim    
    elif op is 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    elif op is 'rmsprop':
        optim = torch.optim.RMSprop(model.parameters(), lr=lr)
        return optim 
    pass 

def get_model(mode, dataset, source):
    if mode not in ['res-fc', 'dense-fc', 'dense-fcn', 'dense-cat']:
        raise NotImplementedError('mode {} is not supported'.format(mode))
    
    if mode == 'res-fc':
        return resnet_fc(dataset, source=source)
    elif mode == 'dense-fc':
        return densenet_fc(dataset, source=source)
    elif mode == 'dense-cat':
        return densenet_cat(dataset, source=source)
    elif mode == 'dense-fcn':
        return densenet_fcn(dataset, source=source)
    

def main():
    pass

if __name__ =='__main__':
    main()