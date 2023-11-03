import random

import cv2
import numpy as np
import torchvision.transforms as transforms

# Image Augmentation

def randomCropWithCenterResized(img, size, center, crop_scale):
    
    center0, center1 = center
    max0 = img.shape[0] - 2 * np.abs(img.shape[0] // 2 - center0) - 1
    max1 = img.shape[1] - 2 * np.abs(img.shape[1] // 2 - center1) - 1
    
    img_tmp = img.copy()
    img_tmp = img_tmp[center0 - max0 // 2:center0 + max0 // 2 + 1, center1 - max1 // 2:center1 + max1 // 2 + 1]
    # random crop
    crop0 = size[0] * crop_scale[0]
    crop1 = size[1] * crop_scale[1]
    img_tmp = transforms.CenterCrop(size=(crop0, crop1))(img_tmp)
    
    # resize
    img_tmp = cv2.resize(img_tmp, size[::-1])
    
    return img_tmp

def randomCropWithCenterInside(img, size, center, crop_scale):
    
    center0, center1 = center
    crop0, crop1 = size[0] * crop_scale[0], size[1] * crop_scale[1]
    dev0, dev1 = random.randint(-crop0 // 2, crop0 // 2), random.randint(-crop1 // 2, crop1 // 2)
    center0, center1 = center0 + dev0, center1 + dev1
    
    img_tmp = img.copy()
    img_tmp = img_tmp[max(center0 - crop0 // 2, 0):min(center0 + crop0 // 2 + 1, img.shape[0]), 
                      max(center1 - crop1 // 2, 0):min(center1 + crop1 // 2 + 1, img.shape[1])]
    
    img_tmp = cv2.resize(img_tmp, size[::-1])
    
    return img_tmp

def gaussianBlur(img, kernel_size, sigma):
        
    img_tmp = img.copy()
    img_tmp = cv2.GaussianBlur(img_tmp, (kernel_size, kernel_size), sigma)
    
    return img_tmp
    
        
    
# Graph Augmentation

def randomDropAttr(X, p):
    
    mask = np.random.binomial(1, p, size=X.shape[1])
    
    if np.sum(mask) == 0:
        mask[np.random.randint(0, X.shape[1])] = 1
    
    Xm = np.copy(X)
    Xm[:, mask == 1] = 0
    
    return Xm

def randomDropEdge(A, p):
    
    mask = np.random.binomial(1, p, size=A.shape)
    
    if np.sum(mask) == 0:
        mask[np.random.randint(0, A.shape[0]), np.random.randint(0, A.shape[1])] = 1
    
    Am = np.copy(A)
    Am[mask == 1] = 0
    
    return Am