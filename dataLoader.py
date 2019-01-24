from augment import augment
import numpy as np 
import scipy.io as sio
import cv2 
import os 
from config import get_config
import shutil
import matplotlib.pyplot as plt

def augment_nyuv2_data(image_dir, depth_dir, dest_image_dir, dest_depth_dir):
    train_count = 50363
    index = 38810

    repeat_times = 1
    for i in range(38812, train_count):
        image = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(i)))
        depth = sio.loadmat(os.path.join(depth_dir, '{}.mat'.format(i)))
        depth = depth['imgDepth2'].astype(np.float32)

        # Repeat serverl times
        for _ in range(repeat_times):
            #aug_img, aug_depth = augment(image, depth)
            dest_image = os.path.join(dest_image_dir, '{}.jpg'.format(index))
            dest_depth = os.path.join(dest_depth_dir, '{}'.format(index))

            cv2.imwrite(dest_image, image)
            np.save(dest_depth, depth)
    
            index = index + 1
        print(i)

def get_train_data(dataset, l):
    image_dir = get_config(dataset, 'train_image_dir')
    depth_dir = get_config(dataset, 'train_depth_dir')
    input_height = int(get_config(dataset, 'input_height'))
    input_weight = int(get_config(dataset, 'input_weight'))

    count = len(l)

    images = np.zeros([count, 3, input_height, input_weight], dtype=np.uint8) 
    depths = np.zeros([count, input_height, input_weight], dtype=np.float32)
    
    for i in range(len(l)):
        image_path = os.path.join(image_dir, '{}.jpg'.format(l[i]))
        depth_path = os.path.join(depth_dir, '{}.npy'.format(l[i]))

        image = cv2.imread(image_path)
        depth = np.load(depth_path)
        
        # image = cv2.resize(image, (input_weight, input_height))
        # depth = cv2.resize(depth, (input_weight, input_height))
        
        # if dataset is 'Make3D':
        #     depth[depth < 0.] = 0.
        #     depth[depth > 70.] = 70.

        image = np.transpose(image,(2, 0, 1))
        images[i] = image
        depths[i] = depth
    
    return images, depths

def get_test_data(dataset, start_index, batch_size):
    source_weight = int(get_config(dataset, 'source_weight'))
    source_height = int(get_config(dataset, 'source_height'))
    input_weight = int(get_config(dataset, 'input_weight'))
    input_height = int(get_config(dataset, 'input_height'))
    depth_dir = get_config(dataset, 'test_image_dir')
    image_dir = get_config(dataset, 'test_depth_dir')
    
    depths = np.zeros([batch_size, source_height, source_weight], dtype = np.float32)
    images = np.zeros([batch_size, 3,input_height, input_weight], dtype = np.uint8)

    for i in range(batch_size):
        image_path = os.path.join(image_dir, '{}.jpg'.format(start_index + i))
        depth_path = os.path.join(depth_dir, '{}.npy'.format(start_index + i))

        image = cv2.imread(image_path)
        image = cv2.resize(image, (input_weight, input_height))
        depth = np.load(depth_path)
        
        if dataset is 'Make3D':
            depth = cv2.resize(depth, (source_weight, source_height))
        image = np.transpose(image, [2, 0, 1])

        images[i] = image
        depths[i] = depth
    
    return images, depths

def get_test_data_with_index(dataset, l):
    image_dir = get_config(dataset, 'test_image_dir')
    depth_dir = get_config(dataset, 'test_depth_dir')
    input_height = int(get_config(dataset, 'input_height'))
    input_weight = int(get_config(dataset, 'input_weight'))

    count = len(l)

    images = np.zeros([count, 3, input_height, input_weight], dtype=np.uint8) 
    depths = np.zeros([count, input_height, input_weight], dtype=np.float32)
    
    for i in range(len(l)):
        image_path = os.path.join(image_dir, '{}.jpg'.format(l[i]))
        depth_path = os.path.join(depth_dir, '{}.npy'.format(l[i]))

        image = cv2.imread(image_path)
        depth = np.load(depth_path)
        
        image = cv2.resize(image, (input_weight, input_height))
        depth = cv2.resize(depth, (input_weight, input_height))

        image = np.transpose(image,(2, 0, 1))
        images[i] = image
        depths[i] = depth
    
    return images, depths


def extract_nyuv2_test_data(image_mat, depth_mat, dest_image_dir, dest_depth_dir):
    images = sio.loadmat(image_mat)
    depths = sio.loadmat(depth_mat)

    images = images['test_images']
    depths = depths['test_depths'].astype(np.float32)
    
    for i in range(654):
        image = images[:, :, :, i]
        depth = depths[:, :, i]

        image_path = os.path.join(dest_image_dir, '{}.jpg'.format(i))
        depth_path = os.path.join(dest_depth_dir, '{}'.format(i))

        cv2.imwrite(image_path, image)
        np.save(depth_path, depth)

def extract_make3d_train_data(source_image, source_depth, dest_image_dir, dest_depth_dir):
    depth_prefix = 'depth_sph_corr'

    index = 1
    for file in os.listdir(source_image):
        image_path = os.path.join(source_image, file)
        file = file[3:-3]
        file = '{}{}mat'.format(depth_prefix, file)
        
        depth_path = os.path.join(source_depth, file)
        try:
            depth = sio.loadmat(depth_path)
        except:
            print (depth_path)
            continue

        depth = depth['Position3DGrid'].astype(np.float32)
        depth = depth[:, :, 3]

        dest_image_path = os.path.join(dest_image_dir, '{}.jpg'.format(index))
        dest_depth_path = os.path.join(dest_depth_dir, '{}'.format(index))
        
        shutil.copy(image_path, dest_image_path)
        np.save(dest_depth_path, depth)
        index = index + 1

def augment_make3d_train_data(image_dir, depth_dir, dest_dir):
    train_count = 400
    index = 1

    repeat_times = 1
    weight = int(get_config('Make3D', 'source_weight'))
    height = int(get_config('Make3D', 'source_height'))

    for i in range(1, train_count + 1):
        image = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(i)))
        depth = np.load(os.path.join(depth_dir, '{}.npy'.format(i)))

        image = cv2.resize(image, (weight, height))
        depth = cv2.resize(depth, (weight, height))
        # Repeat serverl times
        for _ in range(repeat_times):
            #aug_img, aug_depth = augment(image, depth)
            aug_img = image
            aug_depth = depth

            dest_image = os.path.join(dest_dir, '{}.jpg'.format(index))
            dest_depth = os.path.join(dest_dir, '{}'.format(index))
  
            cv2.imwrite(dest_image, aug_img)
            np.save(dest_depth, aug_depth)
            # plt.figure()
            # plt.subplot(1, 4, 1)
            # plt.imshow(image)
            # plt.subplot(1, 4, 2)
            # plt.imshow(depth, cmap='gray')
            # plt.subplot(1, 4, 3)
            # plt.imshow(aug_img)
            # plt.subplot(1, 4, 4)
            # plt.imshow(aug_depth, cmap='gray')

            # plt.show()
            index = index + 1
        print(i)

def extract_make3d_test_data(image_dir, depth_dir, dest_dir):
    test_count = 133

    weight = int(get_config('Make3D', 'source_weight'))
    height = int(get_config('Make3D', 'source_height'))
    
    for i in range(test_count):
        image = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(i + 1)))
        depth = np.load(os.path.join(depth_dir, '{}.npy'.format(i + 1)))
        depth = depth[:, :, 3]

        image = cv2.resize(image, (weight, height))
        depth = cv2.resize(depth, (weight, height))

        dest_image = os.path.join(dest_dir, '{}.jpg'.format(i))
        dest_depth = os.path.join(dest_dir, '{}'.format(i))

        cv2.imwrite(dest_image, image)
        np.save(dest_depth, depth)
        print (i)

def check_file():
    for i in range(50363):
        path = os.path.join(r'train_data\nyuv2\depth', '{}.npy'.format(i))
        if os.path.exists(path) is False:
            print (path)

def move_file():
    index = 0
    input_weight = int(get_config('NyuV2', 'input_weight'))
    input_height = int(get_config('NyuV2', 'input_height'))
    for i in range(50359):
        depth_path = os.path.join(r'D:\nyuv2\train_data\nyuv2\depth', '{}.npy'.format(i))
        image_path = os.path.join(r'D:\nyuv2\train_data\nyuv2\image', '{}.jpg'.format(i))

        image = cv2.imread(image_path)
        depth = np.load(depth_path)

        image_save = image[49:480-13, 42:640-42]
        depth_save = depth[49:480-13, 42:640-42]

        image = cv2.resize(image_save, (input_weight, input_height))
        depth = cv2.resize(depth_save, (input_weight, input_height))

        # plt.figure()
        # plt.subplot(2, 2, 1)
        # plt.imshow(image)
        # plt.subplot(2, 2, 2)
        # plt.imshow(depth)
        # plt.subplot(2, 2, 3)
        # plt.imshow(image_save)
        # plt.subplot(2, 2, 4)
        # plt.imshow(depth_save)
        # plt.show()
       
        dest_depth = os.path.join(r'train_data\nyuv2\depth', '{}.npy'.format(index))
        dest_image = os.path.join(r'train_data\nyuv2\image', '{}.jpg'.format(index))
        cv2.imwrite(dest_image, image)
        np.save(dest_depth, depth)
        index += 1

        if i % 1000 == 0:
            print (i)

def valid_range():
    left, right, top, bottem = 0, 0, 0, 0 
    for i in range(50359):
        depth_path = os.path.join(r'D:\nyuv2\train_data\nyuv2\depth_temp', '{}.npy'.format(i))
        depth = np.load(depth_path)

        temp = depth > 0
        left += np.mean(np.argmax(temp, axis=1))
        right += np.mean(np.argmax(np.fliplr(temp), axis=1))

        top +=  np.mean(np.argmax(temp, axis=0))
        bottem += np.mean(np.argmax(temp[::-1], axis=0))

        if i % 1000 == 0:
            print(i)
    left = left / 50363
    right = right / 50363
    top = top / 50363
    bottem = bottem / 50363
    print (left, right, top, bottem)
    #42, 42, 49, 13


def main():
    image_dir = r'd:\nyuv2\data\image'
    depth_dir = r'd:\nyuv2\data\depth'
    dest_image_dir = r'D:\nyuv2\train_data\nyuv2\image' 
    dest_depth_dir = r'D:\nyuv2\train_data\nyuv2\depth'

    
    #check_file()
    #valid_range()
    #move_file()
    #images, depths = get_train_data('NyuV2', [1, 2, 4, 5, 6])
    # augment_nyuv2_data(image_dir, depth_dir, dest_image_dir, dest_depth_dir)
    #extract_nyuv2_test_data('nyuv2/test_images.mat', 'nyuv2/test_depths.mat', 'nyuv2/test', 'nyuv2/test')
    #extract_make3d_train_data('make3d/Train400Img', 'make3d/Train400Depth', 'make3d/train', 'make3d/train')
    #augment_make3d_train_data('make3d/train/', 'make3d/train/', 'make3d/train/temp')
    # extract_make3d_test_data('make3d/test', 'make3d/test', 'make3d/real_test')
if __name__ == '__main__':
    main()
        