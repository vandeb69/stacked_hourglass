import numpy as np
from skimage import transform
import cv2


def add(image, heat_map, alpha=0.6):
    height = image.shape[0]
    width = image.shape[1]

    img = np.uint8(255 * image)

    # resize heat map
    heat_map_resized = transform.resize(heat_map, (height, width))
    normalized_heat_map = np.uint8(255 * heat_map_resized)

    heat_map_img = cv2.applyColorMap(1-normalized_heat_map, cv2.COLORMAP_JET)
    final_img = cv2.addWeighted(heat_map_img, alpha, img, 1-alpha, 0, dtype=0)

    return final_img


def inp_outp_img(img, gt, outp, index, stack, channel):
    inp = add(image=img[index, :, :, :],
              heat_map=gt[index, stack, :, :, channel])[np.newaxis]
    outp = add(image=img[index, :, :, :],
               heat_map=outp[index, stack, :, :, channel])[np.newaxis]
    res = np.concatenate((inp, outp), axis=2)
    return res


if __name__ == "__main__":
    import os
    from loader import StackedHourglassLoader
    import json
    from easydict import EasyDict

    os.chdir('..')

    with open('config.json', 'r') as f:
        config = json.load(f)
    config = EasyDict(config)

    loader = StackedHourglassLoader(config)

    img, gtmap, l = next(loader.generator())
    img = img[0, :, :, :]
    gtmap = gtmap[0, 0, :, :, 0]

    add(img, gtmap, alpha=0.7)


