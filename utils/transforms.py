import cv2
import numpy as np
import imaug 
import random

class Compose:
    def __init__(self, ops):
        self.ops = ops
    def __call__(self, x):
        for op in self.ops:
            x = op(x)
        return x

class RandomHorizontalFlip(imaug.RandFlipImage):
    def __init__(self):
        super(RandomHorizontalFlip, self).__init__(flip_code=1)

Resize = imaug.ResizeImage 
RandomCrop = imaug.RandCropImage
Normalize = imaug.NormalizeImage 
ToTensor = imaug.ToCHWImage
