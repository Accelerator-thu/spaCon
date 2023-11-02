import random

import cv2
import numpy as np

# Image Augmentation

def randomCrop(img, crop_size):
    """
    Randomly crop the image to the given size.
    """
    h, w = img.shape[:2]
    x, y = np.random.randint(0, h - crop_size[0]), np.random.randint(0, w - crop_size[1])
    return img[x:x+crop_size[0], y:y+crop_size[1]]
    
def randomRotate(img):
    """
    Randomly rotate the image by the given angle.
    """
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    angle = np.random.randint(0, 360)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def randomFlip(img):
    """
    Randomly flip the image.
    """
    flipCode = random.randint(-1, 1)
    return cv2.flip(img, flipCode)

def randomBlur(img, kernel_size):
    """
    Randomly blur the image.
    """
    return cv2.blur(img, (kernel_size, kernel_size))

def colorDistortion(img):
    """
    Randomly distort the color of the image.
    """
    # Randomly shuffle the channels of the image
    img = img[..., np.random.permutation(3)]
    
    # Randomly scale the channels of the image
    scale = np.random.uniform(0.8, 1.2, size=3)
    img[..., 0] *= scale[0]
    img[..., 1] *= scale[1]
    img[..., 2] *= scale[2]
    
    # Randomly shift the channels of the image
    shift = np.random.uniform(-0.2, 0.2, size=3)
    img[..., 0] += shift[0]
    img[..., 1] += shift[1]
    img[..., 2] += shift[2]
    
    # Clip the image to the valid pixel range
    img = np.clip(img, 0, 255)
    
    return img
    
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