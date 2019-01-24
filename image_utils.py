import numpy as np 
import torch 
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F

def convert_gray(image):
    batch, h, w = image.shape[0], image.shape[2], image.shape[3]
    r, g, b = 0.299, 0.587, 0.144
    gray_image = torch.zeros([batch, h, w])

    for i in range(batch):
        gray_image[i] = batch[i][0] * b + batch[i][0] * g + batch[i][0] * r

    return gray_image

def gradient(gray):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2],[-1, 0, 1]], dtype=torch.float, requires_grad=False).cuda()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0],[1, 2, 1]], dtype=torch.float, requires_grad=False).cuda()
    
    gray = torch.unsqueeze(gray, 1)

    sobel_x = torch.unsqueeze(sobel_x, 0)
    sobel_y = torch.unsqueeze(sobel_y, 0)
    sobel_x = torch.unsqueeze(sobel_x, 0)
    sobel_y = torch.unsqueeze(sobel_y, 0)

    graident_x = F.conv2d(gray, sobel_x, padding=1)
    graident_y = F.conv2d(gray, sobel_y, padding=1)

    graident_x = torch.squeeze(graident_x, 1)
    graident_y = torch.squeeze(graident_y, 1)
    #graident = torch.sqrt(graident_x * graident_x + graident_y * graident_y)
    return graident_x, graident_y

    
def convert_gray_temp(image):
    h, w = image.shape[0], image.shape[1]
    r, g, b = 0.299, 0.587, 0.144
    gray_image = np.zeros([h, w])

    gray_image = image[:, :, 0] * b + image[:, :, 1] * g + image[:, :, 2] * r
    
    gray_image = gray_image.astype(np.int32)
    return gray_image


def main():
    image_path = 'Dense_layerv2.jpg'
    image = cv2.imread(image_path)
    gray = convert_gray_temp(image)
    graident_x, graident_y = graident(gray)

    graident_x = graident_x.numpy()
    graident_y = graident_y.numpy()
    pass

if __name__ == '__main__':
    main()