import numpy as np 
import torch 
from densenet import densenet_cat, densenet_fc, densenet_fcn, densenet_fcn_final
from resnet import resnet_fc, resnet_deconv, resnet_deconv_cat, resnet_final, resnet_deconv_sum, resnet_upsample, resnet_up_projection

def get_optim(model, op, lr):
    if op is 'momentum':
        optim = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9)
        return optim    
    elif op is 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
        return optim
    elif op is 'rmsprop':
        optim = torch.optim.RMSprop(model.parameters(), lr=lr)
        return optim 
    pass 

def get_model(mode, dataset, source):
    if mode not in ['res-fc', 'dense-fc', 'dense-fcn', 'dense-cat', 'dense-final',
                     'resnet_deconv', 'resnet_up_project','resnet_deconv_cat', 'res-final', 'resnet_deconv_sum', 'resnet_upsample']:
        raise NotImplementedError('mode {} is not supported'.format(mode))
    
    if mode == 'res-fc':
        return resnet_fc(dataset, source=source)
    elif mode == 'dense-fc':
        return densenet_fc(dataset, source=source)
    elif mode == 'dense-cat':
        return densenet_cat(dataset, growth_rate=16, block_config=[4, 5, 7, 10, 12], last_layer=15,
                 num_init_features=48, bn_size=4, drop_rate=0.2, efficient=True,
                 constrain=False, source=source)
    elif mode == 'dense-fcn':
        return densenet_fcn(dataset, growth_rate=16, block_config=[4, 5, 7, 10, 12], last_layer=15,
                 num_init_features=48, bn_size=4, drop_rate=0.2, efficient=True,
                 constrain=False, source=source)
    elif mode == 'dense-final':
        return densenet_fcn_final(dataset, growth_rate=16, block_config=[4, 5, 7, 10, 12], last_layer=15,
                 num_init_features=48, bn_size=4, drop_rate=0.2, efficient=True,
                 constrain=False, source=source)
    elif mode == 'resnet_deconv':
        return resnet_deconv(dataset, source=source)
    elif mode == 'resnet_deconv_cat':
        return resnet_deconv_cat(dataset, source=source)
    elif mode == 'res-final':
        return resnet_final(dataset,  source=source)
    elif mode == 'resnet_deconv_sum':
        return resnet_deconv_sum(dataset, source=source)
    elif mode == 'resnet_upsample':
        return resnet_upsample(dataset, source=source)
    elif mode == 'resnet_up_project':
        return resnet_up_projection(dataset,source=source)

def main():
    pass

if __name__ =='__main__':
    main()