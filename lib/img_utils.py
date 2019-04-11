""" Utilities module for working with images and building matrices holding
RGB/YCbCr component values."""
import logging
import math
import os
import numpy as np
from skimage.measure import compare_ssim as ssim
from PIL import Image
import cv2


def calc_metrics(orig_filepath, compr_filepath):
    orig_img = cv2.imread(orig_filepath, 1)
    compr_img = cv2.imread(compr_filepath, 1)

    mse = np.mean((orig_img - compr_img) ** 2)
    if mse == 0:
        return 0, 100

    PIXEL_MAX = 255.0
    return mse, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def calc_metrics_luma(orig_filepath, compr_filepath):
    orig = Image.open(orig_filepath)
    compr = Image.open(compr_filepath)

    orig = pil2grayscale(orig)
    compr = pil2grayscale(compr)

    np_orig = np.array(orig)
    np_compr = np.array(compr)

    mse = np.mean((np_orig - np_compr) ** 2)

    if mse == 0:
        return 0, 100

    PIXEL_MAX = 255.0
    return mse, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def calc_metrics_jpeg(orig_filepath):
    orig = Image.open(orig_filepath)
    orig = orig.convert('RGB')
    orig.save('./imgs/jpeg_test.jpg')

    mse_luma, psnr_luma = calc_metrics_luma(orig_filepath, './imgs/jpeg_test.jpg')
    mse, psnr = calc_metrics(orig_filepath, './imgs/jpeg_test.jpg')
    os.remove('./imgs/jpeg_test.jpg')

    return mse, psnr, mse_luma, psnr_luma


def calc_ssim(orig_filepath, compr_filepath):
    orig = Image.open(orig_filepath)
    compr = Image.open(compr_filepath)
    np_orig = np.array(orig)
    np_compr = np.array(compr)

    print(ssim(np_orig, np_compr))


def get_jpeg_size(img_filepath):
    """ Saves the image as a JPEG, retrieves its size and deletes the image."""
    img = Image.open(img_filepath)
    img.save('./imgs/jpeg_test.jpg')
    size = os.path.getsize('./imgs/jpeg_test.jpg')
    os.remove('./imgs/jpeg_test.jpg')

    return size


def get_orig_size(img_filepath):
    img = Image.open(img_filepath)
    return img.size[0] * img.size[1] * 3


def pil2grayscale(img):
    """ Converts a PIL Image to a grayscale image. Returns a numpy matrix."""
    img_new = img.convert('L')
    npim = np.array(img_new)

    return npim


def pil2ycbcr(img):
    """ Converts a PIL Image to the YCbCr representation. Returns a numpy
    matrix.
    """
    img_new = img.convert('YCbCr')
    npim = np.array(img_new)

    return npim


def create_singlecomponent_matrix(np_img, idx):
    """ RGB pixel value is stored as [x][y][rgb] where rgb is a value in
    [0, 1, 2], corresponding to R/G/B. The 3d matrix is simply sliced
    in the dimension."""
    return np_img[:, :, idx]


def construct_24bit_rgb_matrix(rmat, gmat, bmat):
    """ Creates a 2D RGB matrix in the form of
    [r, g, b, r, g, b, ...
     r, g, b, r, g, b, ...]
    from the R/G/B matrices.
    """
    logging.info('Constructing a 24bit RGB matrix from R/G/B components.')
    w = rmat.shape[0]
    h = rmat.shape[1]

    mat = np.zeros((w, h*3), dtype=np.uint8)

    col = 0
    for i in range(w):
        for j in range(0, h*3, 3):
            mat[i][j] = rmat[i][col]
            mat[i][j+1] = gmat[i][col]
            mat[i][j+2] = bmat[i][col]
            col += 1
        col = 0

    return mat


def decompose_24bit_rgb_matrix(mat):
    """ Decomposes a 2D RGB matrix into its R/G/B matrices."""
    logging.info('Decomposing a 2D RGB matrix into its R/G/B components.')
    w = mat.shape[0]
    h = int(mat.shape[1] / 3)

    rmat = np.zeros((w, h), dtype=np.uint8)
    gmat = np.zeros((w, h), dtype=np.uint8)
    bmat = np.zeros((w, h), dtype=np.uint8)

    col = 0
    for i in range(w):
        for j in range(0, mat.shape[1], 3):
            rmat[i][col] += mat[i, j]
            gmat[i][col] += mat[i, j+1]
            bmat[i][col] += mat[i, j+2]
            col += 1
        col = 0

    return rmat, gmat, bmat


def decompose_into_rgb(np_img):
    """ Decomposes a PIL Image stored in a numpy array into its R/G/B
    components. Returns a tuple containing 2D matrices containing red
    component, green component and blue component.
    """
    logging.info('Decomposing an image into RGB components.')
    rmat = create_singlecomponent_matrix(np_img, 0)  # R index
    gmat = create_singlecomponent_matrix(np_img, 1)  # G index
    bmat = create_singlecomponent_matrix(np_img, 2)  # B index

    return rmat, gmat, bmat


def reconstruct_ycbcr_image(y, cb, cr):
    logging.info('Reconstructing an image from its YCbCr components.')
    w = y.shape[0]
    h = y.shape[1]

    mat = np.zeros((w, h, 3), dtype=np.uint8)

    for i in range(w):
        for j in range(h):
            mat[i][j][0] += y[i, j]
            mat[i][j][1] += cb[i, j]
            mat[i][j][2] += cr[i, j]

    return mat

def reconstruct_image(rmat, gmat, bmat):
    """ Reconstructs an image into the array format as expected by the
    pillow library, being
    [[[r, g, b], [r, g, b]]
     [[r, g, b], [r, g, b]]
     ...
    ]
    """
    logging.info('Reconstructing an image from its RGB components.')
    w = rmat.shape[0]
    h = rmat.shape[1]

    mat = np.zeros((w, h, 3), dtype=np.uint8)

    for i in range(w):
        for j in range(h):
            mat[i][j][0] += rmat[i, j]
            mat[i][j][1] += gmat[i, j]
            mat[i][j][2] += bmat[i, j]

    return mat
