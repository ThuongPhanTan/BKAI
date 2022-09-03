import os
import json
import cv2
import random
from multiprocessing import Pool
from tqdm import tqdm
# from augment import distort, perspective

import PIL.ImageOps
import numpy as np
from PIL import Image

from straug.blur import GaussianBlur, DefocusBlur, MotionBlur, GlassBlur, ZoomBlur
from straug.camera import Contrast, Brightness, JpegCompression, Pixelate
from straug.geometry import Rotate, Perspective, Shrink, TranslateX, TranslateY
from straug.noise import GaussianNoise, ShotNoise, ImpulseNoise, SpeckleNoise
from straug.pattern import VGrid, HGrid, Grid, RectGrid, EllipseGrid
from straug.process import Posterize, Solarize, Invert, Equalize, AutoContrast, Sharpness, Color
from straug.warp import Curve, Distort, Stretch
from straug.weather import Fog, Snow, Frost, Rain, Shadow

FOLDER_NAME = '../../Data/data_crop/train'
txt_path = '../../Data/data_crop/train/labels.txt'

def apply_aug(line):
    instance = line.strip().split('\t')
    if len(instance) != 2: 
        return

    path, label = instance
    path = os.path.join(f'{FOLDER_NAME}', path)
    if not os.path.exists(path):
        return

    im = Image.open(path)
    w, h = im.size
    if h < 20: 
        return

    rng = np.random.default_rng()
    ops = [Curve(rng=rng), Rotate(rng=rng), Perspective(rng), Distort(rng), Stretch(rng), Shrink(rng),
           VGrid(rng), HGrid(rng), Grid(rng)]
    ops.extend([GaussianNoise(rng), ShotNoise(rng), ImpulseNoise(rng), SpeckleNoise(rng)])
    ops.extend([GaussianBlur(rng), DefocusBlur(rng), MotionBlur(rng), GlassBlur(rng), ZoomBlur(rng)])
    ops.extend([Contrast(rng), Brightness(rng), JpegCompression(rng), Pixelate(rng)])
    ops.extend([Fog(rng), Snow(rng), Frost(rng), Rain(rng), Shadow(rng)])
    ops.extend(
        [Posterize(rng), Solarize(rng), Invert(rng), Equalize(rng), AutoContrast(rng), Sharpness(rng), Color(rng)])
    mag = random.randrange(-1, 4)
    mag = -1
    gray = random.random() <= 0.1
    op = ops[random.randrange(len(ops))]
    
    try:
        aug_img = op(im, mag=mag)
    except:
        return None
    if gray:
        aug_img = PIL.ImageOps.grayscale(aug_img)
    return aug_img, im, path, label


def main():


    with open(txt_path, 'r', encoding='utf8') as f:
        data = f.readlines()

    with Pool(8) as p:
        results = p.map(apply_aug, data)

    aug_data = []
    original_aug_data = []

    if not os.path.exists(f'{FOLDER_NAME}/images_augment'):
        os.makedirs(f'{FOLDER_NAME}/images_augment')

    for i, res in tqdm(enumerate(results)):
        if res is None:
            continue

        aug_im, im, path, label = res
        path_aug = path.replace('images', 'images_augment')

        print(path_aug)
        aug_im.save(path_aug)

        path_aug = path_aug.split('/',5)[-1]
        path = path.split('/',5)[-1]

        aug_data.append(f'{path_aug}\t{label}')
        original_aug_data.append(f'{path_aug}\t{label}')
        original_aug_data.append(f'{path}\t{label}')


    print(f'Total: {len(aug_data)}')

    with open(f'{FOLDER_NAME}/augmented_labels.txt', 'w+', encoding='utf-8') as f:
        for line in aug_data:
            f.write(line + '\n')

    with open(f'{FOLDER_NAME}/labels_full_original_augment.txt', 'w+', encoding='utf-8') as f:
        for line in original_aug_data:
            f.write(line + '\n')


if __name__ == '__main__':
    main()
