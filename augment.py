import numpy as np 
import cv2 
from imgaug import augmenters as iaa
import scipy.io as sio
import matplotlib.pyplot as plt

def augment(image, depth):
    """ Augment the image as describe is Eigen's parper
        Training pairs Horizontally flip with 0.5 prob,
        Training pairs rotate with angle [-5, 5],
        Training pairs scale [1. 1.5],
        Image mulitpy with a random value between [0.8, 1.2],

        Return:
            Augmented training pairs 

    """ 
    prob = np.random.rand(1)
    flip = 0

    if prob > 0.5:
        flip = 1

    angle = float((np.random.rand(1) - 0.5) * 10) 
    scale_times = float(np.random.rand(1) / 2 + 1.0)

    seq = iaa.Sequential(
        [
            iaa.Affine(
                rotate=(angle),
                scale=(scale_times),
            ),
            iaa.Fliplr(flip), # Horizontally flip with 0.5 prob,
            iaa.Multiply((0.8, 1.2), True),
    ])
    image = seq.augment_image(image)
    
    del seq[-1]
    depth = seq.augment_image(depth)
    depth = depth / scale_times

    return image, depth

def main():
    image_path = r'd:\nyuv2\data\image\1000.jpg'
    depth_path = r'd:\nyuv2\data\depth\1000.mat'
    
    image = cv2.imread(image_path)
    depth = sio.loadmat(depth_path)
    depth = depth['imgDepth2']
    image, depth = augment(image, depth)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(depth)
    plt.show()

if __name__ =='__main__':
    main()