import os

from PIL import Image
import numpy as np
import tensorflow as tf


def read_data(img_dir):
    print("Reading data ...")
    img_paths = [os.path.join(img_dir, img_name) for
                 img_name in os.listdir(img_dir)]

    images = tf.convert_to_tensor([_preprocess(Image.open(img_path),
                                _get_rot(img_path)) for img_path in img_paths])

    labels = tf.convert_to_tensor([0 if "control" in img_path else 1 for
                                   img_path in img_paths])

    groups = tf.convert_to_tensor([int(''.join(filter(str.isdigit, img_path)))
          + (0 if 'control' in img_path else 10000) for img_path in img_paths])
    # TODO: replace 10000 with the number of controls in the data

    print("Read data")
    return images, labels, groups


def _preprocess(image: Image, rot):

    # whiten black pixels
    R, G, B = image.split()
    r = R.load()
    g = G.load()
    b = B.load()
    w, h = image.size
    for i in range(w):
        for j in range(h):
            if r[i, j] == g[i, j] == b[i, j]:
                r[i, j] = g[i, j] = b[i, j] = 255
    image = Image.merge('RGB', (R, G, B))

    # convert to greyscale
    image = image.convert('L')

    # crop to square around spiral (496 x 496)
    width, height = image.size
    diff = abs(width-height)
    crop = 80
    ltrb = [np.floor(diff/2) + crop, crop, np.ceil(diff/2) + crop, crop]  # left, top, right and bottom crop points
    for _ in range(int(rot/90)):
        ltrb = _shift(ltrb)
    ltrb[2] = width - ltrb[2]
    ltrb[3] = height - ltrb[3]
    image = image.crop(ltrb)

    # make the image smaller
    image = image.resize((100, 100))

    # apply Fast Fourier Transform
    image_array = np.array(image)
    image_array = 255 - image_array
    image_array = image_array / np.max(image_array)
    ft = np.fft.ifftshift(image_array)
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)
    ft[48:52, 48:52] = 0  # block out the very high energy component in the center

    # import matplotlib.pyplot as plt
    # plt.set_cmap("gray")
    # plt.subplot(121)
    # plt.imshow(image_array)
    # plt.subplot(122)
    # plt.imshow(abs(ft))
    # plt.show()

    return tf.convert_to_tensor(ft)


def _shift(lst):
    return lst[1:] + [lst[0]]


def _get_rot(path):
    path_min_extension = path[:-5]
    if path_min_extension.endswith('_rot90'):
        return 90
    elif path_min_extension.endswith('_rot180'):
        return 180
    elif path_min_extension.endswith('_rot270'):
        return 270
    else:
        return 0
