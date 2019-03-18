import numpy 
import math
import cv2
import sys


def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    print('mse:', mse)
    if mse == 0:
        return 100

    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def main():
    if len(sys.argv) != 3:
        print('Usage: python calc_metrics.py orig_image compr_image')
        return False

    orig = sys.argv[1]
    compr = sys.argv[2]
    orig_img = cv2.imread(orig, 0)
    compr_img = cv2.imread(compr, 0)

    d = psnr(orig_img, compr_img)
    print('psnr:', d)

if __name__ == '__main__':
    main()
