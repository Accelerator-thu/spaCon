import os
import random
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.draw import polygon
from torch_geometric.data import Data, DataLoader, Dataset
from torch_sparse import coalesce

eucDist = lambda x1, y1, x2, y2: np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
is_inside = lambda x, y, x2, y2: x >= 0 and x < x2 and y >= 0 and y < y2


class spot(object):
    
    def __init__(self, spot_id=-1, grid_x=-1, grid_y=-1, pixel_x=-1, pixel_y=-1, in_tissue=-1):
        # spot_id: unique identifier for each spot
        # grid_x: x-coordinate of spot in grid
        # grid_y: y-coordinate of spot in grid
        # pixel_x: x-coordinate of spot in pixels
        # pixel_y: y-coordinate of spot in pixels
        # in_tissue: 1 if spot is in tissue, 0 if spot is not in tissue, -1 if not meaningful
        self.spot_id = spot_id
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        self.in_tissue = in_tissue
        
    def __str__(self):
        return "spot_id: {}, grid_x: {}, grid_y: {}, pixel_x: {}, pixel_y: {}, in_tissue: {}".format(self.spot_id, self.grid_x, self.grid_y, self.pixel_x, self.pixel_y, self.in_tissue)
    

# class spotNet(object):
    
#     def __init__(self, spots, tissue_position, scaling_factor):
#         # spots: list of spot objects
#         # tissue_position: tissue position in grid
#         # scaling_factor: scaling factor for tissue position
#         self.spots = spots
#         self.tissue_position = tissue_position
#         self.scaling_factor = scaling_factor
        
#     def __str__(self):
#         return "tissue_position: {}, scaling_factor: {}".format(self.tissue_position, self.scaling_factor)


# Masking

def registerSpotWeb(tissue_position, scaling_factor, fig_type='hires'):
    
    assert fig_type in ['hires', 'lowres']
    
    sf = scaling_factor['tissue_' + fig_type + '_scalef']
    
    spot_id = np.array([item[0] for item in tissue_position])
    in_tissue = np.array([item[1] for item in tissue_position])
    grid_x = np.array([item[2] for item in tissue_position])
    grid_y = np.array([item[3] for item in tissue_position])
    pixel_x = np.array([item[4] for item in tissue_position]) * sf
    pixel_y = np.array([item[5] for item in tissue_position]) * sf
    pixel_x, pixel_y = pixel_x.astype(int), pixel_y.astype(int)
    
    max_x = np.max(grid_x)
    max_y = np.max(grid_y)
    
    spots = np.array([[None for i in range(max_y + 1)] for j in range(max_x + 1)], dtype=object)
    # spots shape: (max_x + 1, max_y + 1)
    
    for i in range(len(spot_id)):
        spots[grid_x[i]][grid_y[i]] = spot(spot_id[i], grid_x[i], grid_y[i], pixel_x[i], pixel_y[i], in_tissue[i])
    
    return spots

def computeWebConstants(spotWeb, webType='hex'):
        
    # spotWeb: spotNet object
    
    if webType == 'hex':
        # compute three axes
        # l -> r
        cLR = []
        for i in range(spotWeb.shape[0]):
            cur = int(spotWeb[i][0] is None)
            if spotWeb[i][cur + 2] is not None and cur + 2 < spotWeb.shape[1]:
                cLR.append(eucDist(spotWeb[i][cur].pixel_x, spotWeb[i][cur].pixel_y, 
                                   spotWeb[i][cur + 2].pixel_x, spotWeb[i][cur + 2].pixel_y))
            cur += 2
        cLR = np.median(cLR)

        # tl -> br
        cTLBR = []
        for i in range(spotWeb.shape[0]):
            if spotWeb[i][0] is not None:
                for j in range(spotWeb.shape[1] - 1):
                    if (i + j + 1) % spotWeb.shape[0] > (i + j) % spotWeb.shape[0]:
                        cTLBR.append(eucDist(spotWeb[(i + j) % spotWeb.shape[0]][j].pixel_x, spotWeb[(i + j) % spotWeb.shape[0]][j].pixel_y, 
                                             spotWeb[(i + j + 1) % spotWeb.shape[0]][j + 1].pixel_x, spotWeb[(i + j + 1) % spotWeb.shape[0]][j + 1].pixel_y))
        cTLBR = np.median(cTLBR)
        
        # tr -> bl
        cTRBL = []
        for j in range(spotWeb.shape[1] - 1, 0, -1):
            if spotWeb[0][j] is not None:
                for i in range(spotWeb.shape[0] - 1):
                    if (j - i - 1) % spotWeb.shape[1] < (j - i) % spotWeb.shape[1]:
                        cTRBL.append(eucDist(spotWeb[i][(j - i) % spotWeb.shape[1]].pixel_x, spotWeb[i][(j - i) % spotWeb.shape[1]].pixel_y, 
                                             spotWeb[i + 1][(j - i - 1) % spotWeb.shape[1]].pixel_x, spotWeb[i + 1][(j - i - 1) % spotWeb.shape[1]].pixel_y))
        cTRBL = np.median(cTRBL)
        
        return (cLR, cTLBR, cTRBL)
    

def generateHexMask(side_length, mask_size):
    """
    Generate a mask with a hexagon given the side length.
    
    :param side_length: The length of the side of the hexagon
    :param mask_size: A tuple of (height, width) for the size of the mask
    :return: A 2D NumPy array representing the mask
    """
    # Calculate the radius of the circumscribed circle
    radius = side_length * np.sqrt(3) / 2
    
    # Calculate the center of the mask
    cx, cy = mask_size[1] // 2, mask_size[0] // 2
    
    # Calculate the vertices of the hexagon
    theta = np.pi / 3  # 60 degrees in radians
    vertices = []
    for i in range(6):
        angle = theta * i
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        vertices.append((x, y))
    vertices = np.array(vertices)

    # Create the mask
    rr, cc = polygon(vertices[:, 1], vertices[:, 0], mask_size)
    mask = np.zeros(mask_size, dtype=np.uint8)
    mask[rr, cc] = 1

    return mask


def generateInTissueGrid(spotWeb, img):

    # generate in-tissue grids
    shape = img.shape
    grid = np.zeros((shape[0], shape[1]))
    for i in range(spotWeb.shape[0]):
        for j in range(spotWeb.shape[1]):
            if spotWeb[i][j] is not None and spotWeb[i][j].in_tissue == 1:
                grid[round(spotWeb[i][j].pixel_x)][round(spotWeb[i][j].pixel_y)] = 1
    
    return grid


def generateMaskForHexImg(spotWeb, img):
    
    cLR, cTLBR, cTRBL = computeWebConstants(spotWeb)
    sideLen = np.max([cLR, cTLBR, cTRBL])
    mask_size = (int(2 * sideLen), int(2 * sideLen))
    mask = generateHexMask(sideLen, mask_size)
    
    grid = generateInTissueGrid(spotWeb, img)
    
    # Do a convolution of the grid with the mask
    conv = cv2.filter2D(grid, -1, mask)
    
    return conv


# TBD
# cv2.erode()
# cv2.dilate()

# Downsampling

def downSampleImg2x(img):
    return cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    
def downSampleImg4x(img):
    return cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_AREA)


# Graph Construction

def constructHexGraph(row_idx, col_idx):
    
    rel_idx = [(0, 2), (1, 1), (1, -1)]
    
    # order by tissue position
    
    assert row_idx.shape == col_idx.shape, "row_idx and col_idx must have the same shape"
    
    pos_idx = np.concatenate((row_idx.reshape(-1, 1), col_idx.reshape(-1, 1)), axis=1)
    pos_dict = {}
    for i in range(pos_idx.shape[0]):
        pos_dict[tuple(pos_idx[i])] = i
        
    # construct graph
    edge_idx = []
    for i in range(pos_idx.shape[0]):
        for j in range(6):
            if tuple(pos_idx[i] + rel_idx[j]) in pos_dict:
                edge_idx.append([i, pos_dict[tuple(pos_idx[i] + rel_idx[j])]])
    
    edge_idx = np.array(edge_idx)
    edge_idx = np.concatenate((edge_idx, edge_idx[:, [1, 0]]), axis=0)
    edge_idx = edge_idx.astype(int).T
    
    return edge_idx

def prepareGraphData(visium_data, preprocessed=True, preprocess=None, compress=False, encoder=None):
    
    row_idx = visium_data.obs['array_row'].values
    col_idx = visium_data.obs['array_col'].values
    
    # sparse X
    X = visium_data.X   # in Compressed Sparse Row format
    
    if not preprocessed:
        if preprocess is None:
            raise ValueError("preprocess must be specified if preprocessed is False")
        X = preprocess(X)
    
    if compress:
        if encoder is None:
            raise ValueError("encoder must be specified if compress is True")
        X = encoder(X)
    
    # edge index
    edge_idx = constructHexGraph(row_idx, col_idx)
    
    # construct graph
    G = Data(x=torch.tensor(X.todense(), dtype=torch.float), edge_index=torch.tensor(edge_idx, dtype=torch.long))
    
    