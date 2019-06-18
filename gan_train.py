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

def trainWithGan(mode, dataset, epochs, loss='l1Loss', op='momentum', lr=1e-2, batch_size=4,
                 load_gen_model=None, load_dis_model=None, save_dir=None, source=False, start_index = 0):
    critic = 1
    device = torch.device('cuda:0')
    # Get model 
    modelG = get_model(mode, dataset, source)
    modelG.to(device)

    modelD = Discriminator()
    modelD.to(device)
    
    error_list = []
    #Check if there is trained model
    if load_gen_model is not None:
        modelG.load_state_dict(torch.load(load_gen_model))
    
    if load_dis_model is not None:
        modelD.load_state_dict(torch.load(load_dis_model))

    # traditional loss function 
    loss_normal = get_loss(loss)
    loss_normal.to(device)

    # loss in Gen and Dis 
    loss_G = nn.BCELoss()
    loss_D = nn.BCELoss()
    loss_G.to(device)
    loss_D.to(device)

    optimD = torch.optim.Adam(modelD.parameters(), lr=1e-4,)
    optimG = get_optim(modelG, op, lr)

    train_count = int(get_config(dataset,  'train_count'))
    total_it = math.ceil(train_count * epochs / batch_size)

    epoch = train_count // batch_size

    if save_dir is not None:
        if os.path.exists(save_dir) is False:
            os.mkdir(save_dir)

    real_label = 1
    fake_label = 0

    for i in range(total_it):
        # update the parameters in the DIS
        for c in range(critic):
            batch_list = list(np.random.randint(1, train_count, size=[batch_size]))

            images, depths = get_train_data(dataset, batch_list)
            images = torch.from_numpy(images).cuda().float()
            depths = torch.from_numpy(depths).cuda()

            mask = torch.tensor(depths)
            if dataset is 'Make3D':
                mask = (depths > 0.0) & (depths < 70.0)
            elif dataset is 'NyuV2':
                mask = (depths > 0.0) & (depths < 10.0)

            #fake images
            label = torch.full((batch_size, ), fake_label, device=device)
            modelD.zero_grad()
            
            #Gen the depth predicticon and this is for fake lable
            predict = modelG(images)
            predict_temp = torch.unsqueeze(predict, 1)

            output = modelD(predict_temp.detach())
            output = output.view(batch_size, -1)
            errD_fake = loss_D(output, label)
            errD_fake.backward()
            
            # real images
            output_real = modelD(torch.unsqueeze(depths, 1))
            output_real = output_real.view(batch_size, -1)
            label.fill_(real_label)
            errD_real = loss_D(output_real, label)
            errD_real.backward()  
            optimD.step()

            if i % 10 == 0 and c == 0 :
                # errD = errD_real + errD_fake
                # print ('In iter {}, loss in the dis is {}'.format(i, errD.item()))
                # fake_part = torch.sum(output < 0.5).detach().cpu().numpy() / batch_size
                # real_part = torch.sum(output_real > 0.5).detach().cpu().numpy() / batch_size
                # print('gen depth successce right in D is {}, real depth successce right in D is {}'.format(fake_part, real_part))
                pass
        #updata the parameters in the geneotator
        batch_list = list(np.random.randint(1, train_count, size=[batch_size]))
        images, depths = get_train_data(dataset, batch_list)
        images = torch.from_numpy(images).cuda().float()
        depths = torch.from_numpy(depths).cuda()

        mask = torch.tensor(depths)
        if dataset is 'Make3D':
            mask = (depths > 0.0) & (depths < 70.0)
        elif dataset is 'NyuV2':
            mask = (depths > 0.0) & (depths < 10.0)

        modelG.zero_grad()
        predict = modelG(images)
        label.fill_(real_label)

        input = torch.unsqueeze(predict, 1)
        output = modelD(input)
        g_loss = loss_G(output, label)
        g_normal_loss = loss_normal(predict, depths, mask)   
        errG = g_loss

        errG.backward()
        optimG.step()

        if i % 10 == 0:
            print ('In iter {}, gene loss is {}, normal loss is {}'.format(i, g_loss.item(), g_normal_loss.item()))
            #print (i, g_loss.cpu().detach().numpy(), g_normal_loss.cpu().detach().numpy())
            # #here we check the discrimator performance
            # output = output.detach().cpu().numpy()
            # output = np.sum(output < 0.5)
            # fake_rate = output / batch_size

            # real = modelD(depths)
            # real = real.detach().cpu().numpy()
            # real = np.sum(real > 0.5)
            # real_rate = real / batch_size

            # print ('gen depth successce right in D is {}, real depth successce right in D is {}'.format(fake_rate, real_rate))

        if i % epoch == epoch - 1:
            torch.save(modelG.state_dict(), '{}/{}.pkl'.format(save_dir,start_index + i // epoch))
    pass


def trainWithWganGp(mode, dis_mode,dataset, epochs, loss='l1Loss', op='momentum', lr=1e-2, batch_size=4,
                 load_gen_model=None, load_dis_model=None, save_dir=None, source=False, start_index = 0):

    device = torch.device('cuda:0')
    # Get model 
    modelG = get_model(mode, dataset, source)
    modelG.to(device)

    modelD = get_discriminator(dis_mode, dataset)
    modelD.to(device)
    
    #Check if there is trained model
    if load_gen_model is not None:
        modelG.load_state_dict(torch.load(load_gen_model))
    
    if load_dis_model is not None:
        modelD.load_state_dict(torch.load(load_dis_model))

    modelG.train()
    modelD.train()

    # traditional loss function 
    loss_normal = get_loss(loss)
    loss_normal.to(device)

    # loss in Gen and Dis 
    optimD = torch.optim.Adam(modelD.parameters(), lr=5e-5, betas=(0.5, 0.9))
    optimG = get_optim(modelG, op, lr)

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

        modelD.zero_grad()
        
        #Gen the depth predicticon and this is for fake lable
        predict = modelG(images)
        predict_temp = torch.unsqueeze(predict, 1)

        fake_predict_temp = torch.cat([images, predict_temp.detach()], 1)
        image_depth_pair = torch.cat([images, torch.unsqueeze(depths, 1)], 1)

        #calcuate normal gan loss
        output = modelD(fake_predict_temp.detach())
        output_fake = output.view(batch_size, -1)

        output_real = modelD(image_depth_pair)
        output_real = output_real.view(batch_size, -1)
        
        wganDistance = (output_fake - output_real).mean() 

        gradientPenalty = calcGradientPenalty(modelD, image_depth_pair, fake_predict_temp)

        errD = wganDistance + 10 * gradientPenalty
        errD.backward()
        optimD.step()

        #updata the parameters in the geneotator
        modelG.zero_grad()

        real_predict_temp = torch.cat([images, predict_temp], 1)
        output = modelD(real_predict_temp)
        g_loss = -torch.mean(output)
        # g_normal_loss = loss_normal(predict, depths, mask)   
        errG = g_loss

        errG.backward()
        optimG.step()

        if i % 50 == 0:
            print (i, wganDistance.cpu().detach().numpy())

        if i % epoch == epoch - 1:
            torch.save(modelG.state_dict(), '{}/gen{}.pkl'.format(save_dir,start_index + i // epoch))
            torch.save(modelD.state_dict(), '{}/dis{}.pkl'.format(save_dir,start_index + i // epoch))
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