""" Utilities module for working with images and building matrices holding
RGB/YCbCr component values."""
import logging
import math
import os
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr
from PIL import Image
import cv2


def calc_metrics_jpeg(orig_filepath):
    """ Function calculating PSNR and SSIM of a JPEG image compared to original
    one.
    """
    orig = Image.open(orig_filepath)
    orig = orig.convert('RGB')
    orig.save('./imgs/jpeg_test.jpg')

    psnr = calc_psnr_skimage(orig_filepath, './imgs/jpeg_test.jpg')
    ssim = calc_ssim(orig_filepath, './imgs/jpeg_test.jpg')
    os.remove('./imgs/jpeg_test.jpg')

    return psnr, ssim


def calc_psnr_skimage(orig_filepath, compr_filepath):
    """ Function calculating the PSNR of a compressed image compared to original
    one.
    """
    orig = Image.open(orig_filepath)
    compr = Image.open(compr_filepath)
    np_orig_gray = pil2grayscale(orig)
    np_compr_gray = pil2grayscale(compr)

    return compare_psnr(np_orig_gray, np_compr_gray)


def calc_ssim(orig_filepath, compr_filepath):
    """ Function calculating the SSIM of a compressed image compared to original
    one.
    """
    orig = Image.open(orig_filepath)
    compr = Image.open(compr_filepath)
    np_orig = np.array(pil2grayscale(orig))
    np_compr = np.array(pil2grayscale(compr))

    return ssim(np_orig, np_compr)


def get_jpeg_size(img_filepath):
    """ Saves the image as a JPEG, retrieves its size and deletes the image."""
    img = Image.open(img_filepath)
    img.save('./imgs/jpeg_test.jpg')
    size = os.path.getsize('./imgs/jpeg_test.jpg')
    os.remove('./imgs/jpeg_test.jpg')

    return size


def get_orig_size(img_filepath):
    """ Function returning the size of an RGB 24bit image."""
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
    """ Reconstructs separated YCbCr components into one matrix."""
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
