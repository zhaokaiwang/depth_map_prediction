import numpy as np 
import cv2 
import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt 
from config import get_config
from loss import get_loss, error_mertic
from model import get_optim, get_model
from dataLoader import get_train_data, get_test_data, get_test_data_with_index
import os 
import math
from image_utils import gradient
from gan import Discriminator, WganDiscriminator, calcGradientPenalty
import torchvision
from scipy import io

def write_result(dataset, mode, load_model, index):
    device = torch.device('cuda:0')
    test_count = int(get_config(dataset, 'test_count'))
    images, depths = get_test_data_with_index(dataset, [index])

    images = torch.from_numpy(images)
    images = images.to(device).float()

    model = get_model(mode, dataset, source=False)
    model.load_state_dict(torch.load(load_model))
    model.eval()
    model.to(device)

    output = model(images)
    output = output.detach().cpu().numpy()
    output = np.squeeze(output, 0)
    io.savemat('result.mat', {'result': output})

def visual_and_make_grid(dataset, mode_list, load_model_list, image_count=5):
    """Show a list of results from different models and type of model
    """

    device = torch.device('cuda:0')
    test_count = int(get_config(dataset, 'test_count'))
    indics = np.random.random_integers(1, test_count, size = image_count)
    indics = [1, 10, 100, 1000]
    images, depths = get_test_data_with_index(dataset, indics)

    images = torch.from_numpy(images)
    images = images.to(device).float()
    result_list = [depths]

    for i in range(len(model_list)):
        mode = model_list[i]
        model = get_model(mode, dataset, source=False)
        model.load_state_dict(torch.load(load_model_list[i]))
        model.eval()
        model.to(device)

        output = model(images)
        output = output.detach().cpu().numpy()
        result_list.append(output)

    # cancate to final output 
    pad_result = []
    for depth in result_list:
        depth = np.pad(depth, ((0, 0), (2, 2), (2, 2)), 'constant')
        pad_result.append(depth)

    # concate each row
    final_result = []
    for depth in pad_result:
        temp = depth[0]

        for i in range(1, image_count):
            temp = np.vstack((temp, depth[i]))
        final_result.append(temp)
    
    result = final_result[0]
    for i in range(1, len(mode_list) + 1):
        result = np.hstack((result, final_result[i]))
    
    plt.figure()
    plt.imshow(result)
    plt.show()
    pass

def visual_result(dataset, model, index):
    source_weight = int(get_config(dataset, 'source_weight'))
    source_height = int(get_config(dataset, 'source_height'))
    input_weight = int(get_config(dataset, 'input_weight'))
    input_height = int(get_config(dataset, 'input_height'))

    test_image_dir = get_config(dataset, 'test_image_dir')
    test_depth_dir = get_config(dataset, 'test_depth_dir')

    image_path = os.path.join(test_image_dir, '{}.jpg'.format(index))
    depth_path = os.path.join(test_depth_dir, '{}.npy'.format(index))

    image = cv2.imread(image_path)
    depth = np.load(depth_path)

    input = cv2.resize(image, (input_weight, input_height))
    input = np.transpose(input, (2, 0, 1))
    input = np.expand_dims(input, axis=0)

    input = torch.from_numpy(input).cuda().float()

    predict = model(input)
    predict = predict.cpu().detach().numpy()
    predict = np.squeeze(predict, axis=0)
    #cv2.imwrite('predict{}.jpg'.format(index), predict)
    
    #np.savetxt('predict{}'.format(index), predict)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.imshow(depth)
    plt.subplot(1, 3, 3)
    plt.imshow(predict)
    plt.show()
    # np.save('result/predict{}'.format(index), predict)
    # np.save('result/depth{}'.format(index), depth)
    # cv2.imwrite('result/source{}.jpg'.format(index), image)

def visual_random(mode, dataset, load_model, count):
    device = torch.device('cuda:0')
    
    model = get_model(mode, dataset, source=True)
    model.load_state_dict(torch.load(load_model))
    model.to(device)
    model.eval()

    test_count = int(get_config(dataset, 'test_count'))
    l = np.random.randint(1, test_count, [count])

    for i in range(len(l)):
        index = l[i]
        visual_result(dataset, model, index)
        pass

def visual_list(dataset, mode, load_model, index):
    device = torch.device('cuda:0')
    model = get_model(mode, dataset, source=False)
    model.to(device)

    model.load_state_dict(torch.load(load_model))
    model.eval()

    count = len(index)
   
    images, depths = get_test_data_with_index(dataset, index)
    images_test = torch.from_numpy(images).cuda().float()
    predicts = model(images_test)
    predicts = predicts.cpu().detach().numpy()

    images = np.transpose(images, [0, 2, 3, 1])
    plt.figure()
    for i in range(count):
        plt.subplot(count, 3, i * 3 + 1)
        plt.imshow(images[i])
        plt.axis('off')

        plt.subplot(count, 3, i * 3 + 2)
        plt.imshow(depths[i])
        plt.axis('off')

        plt.subplot(count, 3, i * 3 + 3)
        plt.imshow(predicts[i])
        plt.axis('off')
    plt.subplots_adjust(bottom=.01, top=.99, left=.01, right=.99, wspace=0.0, hspace=0.0)
    plt.show()


def train(mode, dataset, epochs, loss='l1Loss', op='momentum', lr=1e-2, batch_size=4,
          load_model=None, save_dir=None, source=False, start_index = 0, with_grad = False):
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

        loss = torch.tensor([0.0]).float().to(device)

        if isinstance(predict, list):
            for index in range(len(predict)):
                loss = loss + loss_fn(predict[index], depths, mask)
        else: 
            loss = loss_fn(predict, depths, mask)
        
        if with_grad is True:
            grad_loss = get_loss('l2Loss')
            predict_grad_x, predict_grad_y = gradient(predict)
            depth_grad_x, depth_grad_y = gradient(depths)
            loss += grad_loss(predict_grad_x, depth_grad_x, mask) + grad_loss(predict_grad_y, depth_grad_y, mask)
        
        loss.backward()

        optim.step()

        if i % 100 == 0:
            print (i, loss.cpu().detach().numpy())

        if i % epoch == epoch - 1:
            torch.save(model.state_dict(), '{}/{}.pkl'.format(save_dir,start_index + i // epoch))
    pass

def test(dataset, mode,  load_model, batch_size=4):
    device = torch.device('cuda:0')
    test_count = int(get_config(dataset, 'test_count'))
    
    model = get_model(mode, dataset, source=True)
    model.to(device)

    model.load_state_dict(torch.load(load_model))
    model.eval()

    temp = test_count // batch_size

    thrLoss  = torch.Tensor([0.0]).long().cuda()
    absLoss  = torch.Tensor([0.0]).float().cuda()
    sqrLoss  = torch.Tensor([0.0]).float().cuda()
    rmsLinLoss = torch.Tensor([0.0]).float().cuda()
    rmsLogLoss = torch.Tensor([0.0]).float().cuda()
    log10Loss = torch.Tensor([0.0]).float().cuda()
    
    count = torch.Tensor([0.]).long().cuda()

    #Get test data
    for i in range(temp):
        images, depths = get_test_data(dataset, i * batch_size, batch_size)
        images = torch.from_numpy (images).cuda().float()

        depths = torch.from_numpy(depths).cuda()
        predict = model(images)

        if dataset is 'NyuV2':
            mask = (depths > 0.) & (depths < 10.) 

            predict = predict[mask]
            depths = depths[mask]
            count += torch.sum(mask)
        elif dataset is 'Make3D':
            mask = (depths > 0.) & (depths < 70.0) 

            predict = predict[mask]
            depths = depths[mask]
            count += torch.sum(mask)

        thr, abs, sqr, rmsLin, rmsLog, log10 = error_mertic(predict, depths)

        thrLoss = thrLoss + thr.detach()
        absLoss = absLoss + abs.detach()
        sqrLoss = sqrLoss + sqr.detach()
        rmsLinLoss = rmsLinLoss + rmsLin.detach()
        rmsLogLoss = rmsLogLoss + rmsLog.detach()
        log10Loss = log10Loss + log10.detach()
    
    last = test_count % batch_size
    if last is not 0:
        images, depths = get_test_data(dataset, temp * batch_size, last)
        images = torch.from_numpy (images).cuda().float()
        depths = torch.from_numpy(depths).cuda()

        predict = model(images)

        if dataset is 'NyuV2':
            mask = (depths > 0) & (depths < 10.) 

            predict = predict[mask]
            depths = depths[mask]
            count += torch.sum(mask)
        else:
            mask = (depths > 0) & (depths < 70.) 

            predict = predict[mask]
            depths = depths[mask]
            count += torch.sum(mask)

        thr, abs, sqr, rmsLin, rmsLog, log10 = error_mertic(predict, depths)

        thrLoss = thrLoss + thr.detach()
        absLoss = absLoss + abs.detach()
        sqrLoss = sqrLoss + sqr.detach()
        rmsLinLoss = rmsLinLoss + rmsLin.detach()
        rmsLogLoss = rmsLogLoss + rmsLog.detach()
        log10Loss = log10Loss + log10.detach()

    count = count.float()
    thrLoss = thrLoss.float() / count

    absLoss = absLoss / count
    sqrLoss = sqrLoss / count
    rmsLinLoss = torch.sqrt(rmsLinLoss / count)
    rmsLogLoss = torch.sqrt(rmsLogLoss / count)
    log10Loss = log10Loss / count

    thrLoss = thrLoss.cpu().detach().numpy()
    absLoss = absLoss.cpu().detach().numpy()
    sqrLoss = sqrLoss.cpu().detach().numpy()
    rmsLinLoss = rmsLinLoss.cpu().detach().numpy()
    rmsLogLoss = rmsLogLoss.cpu().detach().numpy()
    log10Loss = log10Loss.cpu().detach().numpy()

    print (thrLoss, absLoss, sqrLoss, rmsLinLoss, rmsLogLoss, log10Loss) 
    pass

def trainWithGan(mode, dataset, epochs, loss='l1Loss', op='momentum', lr=1e-2, batch_size=4,
                 load_gen_model=None, load_dis_model=None, save_dir=None, source=False, start_index = 0):
    critic = 5
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

    modelG.train()
    modelD.train()

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
        for _ in range(critic):
            batch_list = list(np.random.randint(1, train_count, size=[batch_size]))

            images, depths = get_train_data(dataset, batch_list)
            images = torch.from_numpy(images).cuda().float()
            depths = torch.from_numpy(depths).cuda()

            mask = torch.tensor(depths)
            if dataset is 'Make3D':
                mask = (depths > 0.0) & (depths < 70.0)
            elif dataset is 'NyuV2':
                mask = (depths > 0.0) & (depths < 10.0)

            label = torch.full((batch_size, ), fake_label, device=device)
            modelD.zero_grad()
            
            #Gen the depth predicticon and this is for fake lable
            predict = modelG(images)
            predict_temp = torch.unsqueeze(predict, 1)

            output = modelD(predict_temp.detach())
            output = output.view(batch_size, -1)
            errD_fake = loss_D(output, label)
            errD_fake.backward()

            output_real = modelD(torch.unsqueeze(depths, 1))
            output_real = output_real.view(batch_size, -1)
            label.fill_(real_label)
            errD_real = loss_D(output_real, label)
            errD_real.backward()  

            if i % 100 == 0:
                errD = errD_real + errD_fake
                print ('In iter {}, loss in the dis is {}'.format(i, errD.item()))
            optimD.step()

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

        output = modelD(predict)
        g_loss = loss_G(output, label)
        g_normal_loss = loss_normal(predict, depths, mask)   
        errG = 1e-3 * g_loss + g_normal_loss

        errG.backward()
        optimG.step()

        if i % 100 == 0:
            print ('In iter {}, gene loss is {}, normal loss is {}'.format(i, g_loss.item(), g_normal_loss.item()))
            print (i, g_loss.cpu().detach().numpy(), g_normal_loss.cpu().detach().numpy())

        if i % epoch == epoch - 1:
            torch.save(modelG.state_dict(), '{}/{}.pkl'.format(save_dir,start_index + i // epoch))
    pass


def trainWithWganGp(mode, dataset, epochs, loss='l1Loss', op='momentum', lr=1e-2, batch_size=4,
                 load_gen_model=None, load_dis_model=None, save_dir=None, source=False, start_index = 0):

    device = torch.device('cuda:0')
    # Get model 
    modelG = get_model(mode, dataset, source)
    modelG.to(device)

    modelD = WganDiscriminator()
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

        output = modelD(predict_temp.detach())
        output_fake = output.view(batch_size, -1)

        output_real = modelD(torch.unsqueeze(depths, 1))
        output_real = output_real.view(batch_size, -1)
        
        wganDistance = (output_fake - output_real).mean() 
        gradientPenalty = calcGradientPenalty(modelD, depths, predict_temp.detach())

        errD = wganDistance + 10 * gradientPenalty
        errD.backward()
        optimD.step()

        #updata the parameters in the geneotator
        modelG.zero_grad()

        output = modelD(predict_temp)
        g_loss = -output.mean()
        g_normal_loss = loss_normal(predict, depths, mask)   
        errG = 1e-3 * g_loss + g_normal_loss

        errG.backward()
        optimG.step()

        if i % 100 == 0:
            print (i, g_loss.cpu().detach().numpy(), g_normal_loss.cpu().detach().numpy())

        if i % epoch == epoch - 1:
            torch.save(modelG.state_dict(), 'gen{}/{}.pkl'.format(save_dir,start_index + i // epoch))
            torch.save(modelD.state_dict(), 'dis{}/{}.pkl'.format(save_dir,start_index + i // epoch))
    pass

def write_all_result(dataset, mode, load_model, prefix):
    device = torch.device('cuda:0')
    test_count = int(get_config(dataset, 'test_count'))

    height = int(get_config(dataset, 'input_height'))
    weight = int(get_config(dataset, 'input_weight'))
    predict = np.zeros((test_count, height, weight))
    depths = np.zeros((test_count, height, weight))
    image_set = np.zeros((test_count, height, weight, 3))

    model = get_model(mode, dataset, source=False)
    model.load_state_dict(torch.load(load_model))
    model.eval()
    model.to(device)

    for i in range(0, test_count):
        images, depth = get_test_data_with_index(dataset, [i])
        temp = images

        images = torch.from_numpy(images)
        images = images.to(device).float()

        output = model(images)
        output = output.detach().cpu().numpy()
        output = np.squeeze(output, 0)
        predict[i] = output
        depths[i] = depth

        temp = np.squeeze(temp, axis=0)
        temp = np.transpose(temp, (1, 2, 0))
        image_set[i] = temp
        print(i)

    io.savemat('{}.mat'.format(prefix), {'pred': predict})
    # io.savemat('{}_image.mat'.format(prefix), {'image': image_set})
    # io.savemat('{}_depth.mat'.format(prefix), {'pred': depths})

# trainWithGan('resnet_deconv_cat', 'NyuV2', 10, batch_size=8, loss='l1Loss', 
#       save_dir=r'D:\nyuv2\model\nyuv2\res_deconv_cat_l1_gan', 
#       lr=1e-3 , op='adam', start_index=0)

# trainWithGan('resnet_deconv_cat', 'Make3D', 20, batch_size=8, loss='l1Loss', 
#       save_dir=r'D:\nyuv2\model\make3d\res_deconv_cat_l1_gan', load_dis_model=None, 
#       load_gen_model=None, lr=1e-2 , op='momentum', start_index=0)

# train('resnet_deconv_cat', 'NyuV2', 10, batch_size=8, loss='l1Loss', 
#       save_dir=r'D:\nyuv2\model\nyuv2\res_deconv_cat_new', load_model=None,
#       lr=1e-2, op='momentum', start_index=0, with_grad=False)     

# test('NyuV2', 'resnet_deconv_cat', r'D:\nyuv2\model\nyuv2\res_deconv_cat_l2\9.pkl', 4)
# test('NyuV2', 'resnet_deconv_cat', r'D:\nyuv2\model\nyuv2\res_deconv_cat_berhu\8.pkl', 4)
# test('NyuV2', 'resnet_deconv_cat', r'D:\nyuv2\model\nyuv2\res_deconv_cat_grad\7.pkl', 4)
# test('NyuV2', 'res-fc', r'D:\nyuv2\model\nyuv2\res_fc\5.pkl', 4)
# for i in range(10):
#     print (i)
    
    #test('NyuV2', 'resnet_deconv', r'D:\nyuv2\model\nyuv2\res_deconv\{}.pkl'.format(i), 4)
    # test('NyuV2', 'resnet_deconv_cat', r'D:\nyuv2\model\nyuv2\res_deconv_cat_berhu\{}.pkl'.format(i), 4)
    # test('NyuV2', 'resnet_deconv_cat', r'D:\nyuv2\model\nyuv2\res_deconv_cat_l2\{}.pkl'.format(i), 4)
    # test('NyuV2', 'resnet_deconv_cat', r'D:\nyuv2\model\nyuv2\res_deconv_cat_grad\{}.pkl'.format(i), 4)
    # test('NyuV2', 'resnet_deconv_cat', r'D:\nyuv2\model\nyuv2\res_deconv_cat_new\{}.pkl'.format(i), 4)
    # test('Make3D', 'resnet_deconv_cat', r'D:\nyuv2\model\make3d\res_deconv_cat\{}.pkl'.format(i), 4)
    # test('Make3D', 'resnet_deconv_cat', r'D:\nyuv2\model\make3d\res_deconv_cat_grad\{}.pkl'.format(i), 4)
#     test('Make3D', 'resnet_deconv', r'D:\nyuv2\model\make3d\res_deconv\{}.pkl'.format(i), 4)

# model_list = ['resnet_deconv_cat']
# load_model_list = [r'D:\nyuv2\model\make3d\res_deconv_cat\19.pkl',
#      r'D:\nyuv2\model\make3d\res_deconv_cat_grad\19.pkl']
# while 1:
#     visual_and_make_grid('Make3D', model_list, load_model_list, image_count=2)
#visual_random('resnet_deconv_sum', 'Make3D', r'D:\nyuv2\model\make3d\res_deconv_sum\9.pkl', 5)
#write_result('NyuV2', 'resnet_deconv_cat', r'D:\nyuv2\model\nyuv2\res_deconv_cat\7.pkl', 1)

#io.savemat('depth.mat', {'depth':data})
# write_all_result('NyuV2', 'res-fc', r'D:\nyuv2\model\nyuv2\res_fc\5.pkl', 'nyuv2_fc')
laina = np.load('final.npy').astype(np.float32)
io.savemat('laina.mat', {'pred':laina})