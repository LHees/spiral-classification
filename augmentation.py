import os.path
import shutil
import sys

from PIL import Image


def prepare(img_dir):
    aug_dir = img_dir + '_aug'
    if os.path.exists(aug_dir):
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)

    dir_ls = os.listdir(img_dir)
    for img_name in dir_ls:
        img = Image.open(os.path.join(img_dir, img_name))
        out_path = os.path.join(aug_dir, img_name)
        img.save(out_path)
        for i, aug in enumerate(_augment(img)):
            aug.save(out_path[:-5] + f'_rot{(i+1) * 90}.tiff')


def _augment(image):
    im_rot90 = image.rotate(90, expand=True)
    im_rot180 = image.rotate(180)
    im_rot270 = image.rotate(270, expand=True)
    return im_rot90, im_rot180, im_rot270


if __name__ == "__main__":
    if len(sys.argv) == 2:
        prepare(sys.argv[1])
    else:
        raise ValueError('Usage: python augmentation.py <directory>')
