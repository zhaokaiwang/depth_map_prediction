import torch
import torch.nn as nn 
import torch.functional as F 
import os 
import numpy as np 
from config import get_config

class Discriminator(nn.Module):
    def __init__(self, dataset):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(4, 64, 3, 1, bias=False),
            nn.LeakyReLU(True),

            nn.Conv2d(64, 128, 3, 2 , bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),

            nn.Conv2d(128, 128, 3, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),

            nn.Conv2d(128, 256, 3, 2 , bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),

            nn.Conv2d(256, 256, 3, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),

            nn.Conv2d(256, 512, 3, 2 , bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True),

            nn.Conv2d(512, 512, 3, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True),

            nn.Conv2d(512, 512, 3, 2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True),

            nn.Conv2d(512, 8, 3, 2, bias=False),
            nn.LeakyReLU(True),
        )
        
        if dataset is "NyuV2":
            self.linear = nn.Sequential( 
            nn.Linear(240, 1, bias=False),)
        elif dataset is "Make3D":
            self.linear = nn.Sequential( 
            nn.Linear(192, 1, bias=False),)
        else:
            raise ValueError("dataset is not in valid daset")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)            
        
    def forward(self, input):
        output = self.main(input)
        output = output.view(output.shape[0], -1)
        output = self.linear(output)

        return output

class DiscriminatorWithoutBN(nn.Module):
    def __init__(self, dataset):
        super(DiscriminatorWithoutBN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(4, 64, 3, 1, bias=False),
            nn.LeakyReLU(True),

            nn.Conv2d(64, 128, 3, 2 , bias=False),
            nn.LeakyReLU(True),

            nn.Conv2d(128, 128, 3, 1, bias=False),
            nn.LeakyReLU(True),

            nn.Conv2d(128, 256, 3, 2 , bias=False),
            nn.LeakyReLU(True),

            nn.Conv2d(256, 256, 3, 1, bias=False),
            nn.LeakyReLU(True),

            nn.Conv2d(256, 512, 3, 2 , bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True),

            nn.Conv2d(512, 512, 3, 1, bias=False),
            nn.LeakyReLU(True),

            nn.Conv2d(512, 512, 3, 2, bias=False),
            nn.LeakyReLU(True),

            nn.Conv2d(512, 8, 3, 2, bias=False),
            nn.LeakyReLU(True),
        )
        
        if dataset is "NyuV2":
            self.linear = nn.Sequential( 
            nn.Linear(240, 1, bias=False),)
        elif dataset is "Make3D":
            self.linear = nn.Sequential( 
            nn.Linear(192, 1, bias=False),)
        else:
            raise ValueError("dataset is not in valid daset")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)            
        
    def forward(self, input):
        output = self.main(input)
        output = output.view(output.shape[0], -1)
        output = self.linear(output)

        return output

class WganDiscriminator(nn.Module):
    def __init__(self):
        super(WganDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 48, 3, 2, bias=False),
            nn.ReLU(True),

            nn.Conv2d(48, 96, 3, 2, bias=False),
            nn.ReLU(True),

            nn.Conv2d(96, 192, 3, 2, bias=False),
            nn.ReLU(True),

            nn.Conv2d(192, 384, 3, 2, bias=False),
            nn.ReLU(True),

            nn.Conv2d(384, 768, 3, 2, bias=False),
            nn.ReLU(True),
        )
        self.linear = nn.Sequential( 
            nn.Linear(26880, 1, bias=False),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
        
    def forward(self, input, depth):
        output = self.main(input)
        output = output.view(output.shape[0], -1)
        output = self.linear(output)

        return output 

# patchgan discriminator
class pix2pixelGanDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf = 64, n_layers = 3, norm_layer = nn.BatchNorm2d):
        super(pix2pixelGanDiscriminator, self).__init__()
        use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size = 4, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1

        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** i, 8) # max filter count less than 8
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size = kw, stride = 2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size = kw, stride = 1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
        ]
        
        # last conv
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride = 1, padding=padw)]
        self.model = nn.Sequential(*sequence)
    
    def forward(self, input):
        return self.model(input)


def calcGradientPenalty(netD, real_data, fake_data):
    batch_size = real_data.shape[0]
    temp = torch.rand(batch_size, 1)
    temp = temp.expand(batch_size, 4 * 256 * 192).contiguous().view(batch_size, 4, 256, 192).cuda()

    input = real_data * temp + fake_data * (1. - temp)
    input = input.requires_grad_(True)
    output = netD(input)

    grad = torch.autograd.grad(outputs=output, inputs=input,
        grad_outputs=torch.ones(output.size()).cuda(), create_graph=True,
        retain_graph=True, only_inputs=True)[0]

    gradientPenalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()

    return gradientPenalty 

def get_discriminator(name, dataset):
    if name not in ['gan', 'ganWithoutBN', 'patchGan']:
        raise ValueError("{} not int discriminator set ".format(name))

    if name is "gan":
        return Discriminator(dataset)
    elif name is "ganWithoutBN":
        return DiscriminatorWithoutBN(dataset)
    elif name is "patchGan":
        return pix2pixelGanDiscriminator(4)
    
def main():
    #evice = torch.device('cuda:0')
    netD = pix2pixelGanDiscriminator(4)
    netD.train()
    #netD.to(device)
    
    input = torch.rand(64, 4, 256, 192)
    output = netD(input)
    print(netD)
    print (output.shape)
    pass

if __name__ == "__main__":
    main()