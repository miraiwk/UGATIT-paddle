import cv2
import numpy as np
import sys
import os

def concat_picture(filenames):
    print(f'concat {filenames}')
    out_fname = 'output.png'
    assert not os.path.exists(out_fname)
    # a list of (H, W, C)
    imgs = [cv2.imread(fname) for fname in filenames]
    img = np.concatenate(imgs, 1)
    print(f"result is {out_fname}")
    cv2.imwrite(out_fname, img)

if __name__ == '__main__':
    concat_picture(sys.argv[1:])
