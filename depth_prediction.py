import numpy as np 
import cv2 
import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt 
from config import get_config
from loss import get_loss, error_mertic
from model import get_optim, get_model
from dataLoader import get_train_data, get_test_data
import os 
import math

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
    plt.imshow(depth, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(predict, cmap='gray')
    plt.show()

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

def train(mode, dataset, epochs, loss='l1Loss', op='momentum', lr=1e-2, batch_size=4,
          load_model=None, save_dir=None, source=False, start_index = 0):
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
        mask = depths > 0.0

        if dataset is 'Make3D':
            mask = (depths > 0.0) & (depths < 70.0)
            
        loss = loss_fn(predict, depths, mask)
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

# train('res-fc', 'Make3D', 20, batch_size=8, loss='l1Loss', save_dir='res-fc')
# train('dense-fc', 'Make3D', 5, batch_size=16, loss='l1Loss', save_dir='dense-fc')
# train('dense-cat', 'Make3D', 20, batch_size=8, loss='l1Loss', save_dir='dense-cat')
train('dense-fcn', 'Make3D', 20, batch_size=8, loss='l2Loss', save_dir='dense-fcn/l2Loss', load_model=None, start_index=0, lr=1e-4)

# test('Make3D', 'dense-cat', 'dense-cat/9.pkl', 4)
# for i in range(28, 39):
#     test('Make3D', 'dense-fcn', 'dense-fcn/{}.pkl'.format(i), 4)

# visual_random('dense-cat', 'Make3D', 'dense-fcn/30.pkl', 1)
# visual_random('dense-fcn', 'Make3D', 'dense-fcn/30.pkl', 5)
