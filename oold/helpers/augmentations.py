#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:11:14 2019

@author: aneesh
"""

import cv2
import random
import numpy as np

class RandomHorizontallyFlip(object):
    '''
    Horizontally flips the image and corresponding mask randomly with a given probability

    Parameters:
        p (float) -- probability of the image being flipped. Default value is 0.5.
    '''
    def __init__(self, p):
        self.p = p
    
    def __call__(self, hsi, mask):
        if random.random() < self.p:
            # print(hsi.shape, mask.shape)#(64, 0, 2000) (64, 64, 3) --yi gai zheng wei->(64, 64, 63) (64, 64, 3)
            return (cv2.flip(hsi, 0), cv2.flip(mask, 0))
        return hsi, mask

class RandomVerticallyFlip(object):
    '''
    Vertically flips the image and corresponding mask randomly with a given probability

    Parameters:
        p (float) -- probability of the image being flipped. Default value is 0.5.
    '''
    def __init__(self, p):
        self.p = p
    
    def __call__(self, hsi, mask):
        if random.random() < self.p:
            return (cv2.flip(hsi, 1), cv2.flip(mask, 1))
        return hsi, mask

class RandomTranspose(object):
    '''
    Shifts the image and corresponding mask by 90 deg randomly with a given probability

    Parameters:
        p (float) -- probability of the image being flipped. Default value is 0.5.
    '''
    def __init__(self, p):
        self.p = p
    
    def __call__(self,hsi, mask):
        if random.random() < self.p:
            return (np.transpose(hsi, (1, 0, 2)), cv2.transpose(mask))
        return hsi, mask

class Compose(object):
    '''
    Composes all the transforms together
    
    Parameters:
        augmentations -- list of all augmentations to compose
    '''
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, hsi, mask):
        for a in self.augmentations:
            hsi, mask = a(hsi, mask)

        return hsi, mask