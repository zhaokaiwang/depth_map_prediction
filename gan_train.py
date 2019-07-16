import torch
import torch.nn as nn 
import numpy as np 
import os 
from model import get_model, get_optim
from gan import calcGradientPenalty, get_discriminator
from loss import get_loss
from dataLoader import get_train_data
from config import get_config
import math
import torchvision

def set_required_grad(nets, requireds_grad = False):
    for param in nets.parameters():
        param.requireds_grad = requireds_grad
    
def trainWithGan(mode, dis_mode,dataset, epochs, loss='l1Loss', gan_loss = 'vanllia', op='momentum', lr=1e-2, batch_size=4,
                 load_gen_model=None, load_dis_model=None, save_dir=None, source=False, start_index = 0, LAMBDA = 1):
    device = torch.device('cuda:0')
    # Get model 
    netG = get_model(mode, dataset, source)
    netG.to(device)

    netD = get_discriminator(dis_mode, dataset)
    netD.to(device)
    
    #Check if there is trained model
    if load_gen_model is not None:
        netG.load_state_dict(torch.load(load_gen_model))
    
    if load_dis_model is not None:
        netD.load_state_dict(torch.load(load_dis_model))

    netG.train()
    netD.train()

    # traditional loss function 
    loss_normal = get_loss(loss)
    loss_normal.to(device)

    # gan loss
    loss_gan = get_loss('lsganLoss')
    loss_gan.to(device)

    # loss in Gen and Dis 
    optimD = torch.optim.Adam(netD.parameters(), lr=5e-5, betas=(0.5, 0.999))
    optimG = get_optim(netG, op, lr)

    train_count = int(get_config(dataset,  'train_count'))
    total_it = math.ceil(train_count * epochs / batch_size)

    epoch = train_count // batch_size

    if save_dir is not None:
        if os.path.exists(save_dir) is False:
            os.mkdir(save_dir)

    for i in range(total_it):
        batch_list = list(np.random.randint(1, train_count, size=[batch_size]))

        images, depths = get_train_data(dataset, batch_list)
        images = torch.from_numpy(images).cuda().float()
        depths = torch.from_numpy(depths).cuda()

        mask = torch.tensor(depths)
        if dataset is 'Make3D':
            mask = (depths > 0.0) & (depths < 70.0)
        elif dataset is 'NyuV2':
            mask = (depths > 0.0) & (depths < 10.0)

        #Gen the depth predicticon and this is for fake lable
        fake_predict = netG(images)
        fake_predict = torch.unsqueeze(fake_predict, 1)        

        set_required_grad(netD, True)
        optimD.zero_grad()

        # backward the netD
        fake_predict_temp = torch.cat([images, fake_predict.detach()], 1)
        image_depth_pair = torch.cat([images, torch.unsqueeze(depths, 1)], 1)
        
        #fake
        pred_fake = netD(fake_predict_temp.detach())
        loss_D_fake = loss_gan(fake_predict, 0.)

        #real
        real_predict = netD(image_depth_pair)
        loss_D_real = loss_gan(real_predict, 1.)
        
        lossD = (loss_D_fake + loss_D_real) * 0.5
        lossD.backward(retain_graph=True)
        optimD.step()

        # backward netG
        set_required_grad(netD, False)
        optimG.zero_grad()

        fake_predict_temp = torch.cat([images, fake_predict], 1)
        pred_fake = netD(fake_predict_temp)
        loss_G_GAN = loss_gan(pred_fake, 1.)
        loss_l1 = loss_normal(torch.squeeze(fake_predict, 1), depths, mask)
        loss_G = loss_G_GAN + loss_l1 * LAMBDA 
        loss_G.backward()
        optimG.step()

        if i % 100 == 0:
            print (i, loss_G_GAN.cpu().detach().numpy(), loss_l1.cpu().detach().numpy())

        if i % epoch == epoch - 1:
            torch.save(netG.state_dict(), '{}/gen{}.pkl'.format(save_dir,start_index + i // epoch))
            torch.save(netD.state_dict(), '{}/dis{}.pkl'.format(save_dir,start_index + i // epoch))
    pass

def train_with_perceptual_loss(mode, dataset, epochs, loss='l1Loss', op='momentum', lr=1e-2, batch_size=4,
          load_model=None, save_dir=None, source=False, start_index = 0, with_grad = False,):
    device = torch.device('cuda:0')
    # Get model 
    model = get_model(mode, dataset, source)
    model.to(device)
    
    #Check if there is trained model
    if load_model is not None:
        model.load_state_dict(torch.load(load_model))

    model.train()
    loss_fn = get_loss(loss)
    loss_fn.to(device)
    optim = get_optim(model, op, lr)

    train_count = int(get_config(dataset,  'train_count'))
    total_it = math.ceil(train_count * epochs / batch_size)

    epoch = train_count // batch_size
    LAMBDA = 1
    cnn = torchvision.models.vgg19(pretrained=True).features.to(device).eval()

    if save_dir is not None:
        if os.path.exists(save_dir) is False:
            os.mkdir(save_dir)

    for i in range(total_it):
        batch_list = list(np.random.randint(1, train_count, size=[batch_size]))

        images, depths = get_train_data(dataset, batch_list)
        images = torch.from_numpy(images).cuda().float()
        depths = torch.from_numpy(depths).cuda()

        optim.zero_grad()

        predict = model(images)
        mask = torch.tensor(depths)

        if dataset is 'Make3D':
            mask = (depths > 0.0) & (depths < 70.0)
        elif dataset is 'NyuV2':
            mask = (depths > 0.0) & (depths < 10.0)

        # per-pixel loss
        loss = torch.tensor([0.0]).float().to(device)
        pixel_loss = loss_fn(predict, depths, mask)
        
        if dataset is 'Make3D':
            depths = depths.clamp_(0., 70.)
            depths = depths / 70.
            predict = predict.clamp_(0., 70.)
            predict = predict / 70.
        elif dataset is 'NyuV2':
            depths = depths.clamp_(0., 10.)
            depths = depths / 10.
            predict = predict.clamp_(0., 10.)
            predict = predict / 10.

        # prepare for perceputal loss
        perceptual_loss_fn = get_loss('perceptualLoss')

        content = torch.stack([depths, depths, depths], dim = 1)
        perceptual_model, content_losses, style_losses = perceptual_loss_fn(cnn, content, content)

        input_perceptual =  torch.stack([predict, predict, predict], dim = 1)
        perceptual_model(input_perceptual)

        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.loss
        
        for cl in content_losses:
            content_score += cl.loss

        perceptual_loss = content_score + style_score
        loss =  pixel_loss + LAMBDA * perceptual_loss
        loss.backward()

        optim.step()

        if i % 100 == 0:
            message = 'Epoch [{}/{}]: iter {}: per-pixel loss is {}, features loss is {}'.format(
                i // epoch, epochs, i, pixel_loss.detach().cpu().item(), 
                perceptual_loss.detach().cpu().item())
            print (message)

        if i % epoch == epoch - 1:
            torch.save(model.state_dict(), '{}/{}.pkl'.format(save_dir,start_index + i // epoch))
    pass

def main():
    pass 
    
if __name__ == "__main__":
    pass 