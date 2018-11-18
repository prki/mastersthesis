""" Notes: the script does not check the validity of the file/image header.
          if the file is corrupt OR the header tries to read more than the
          image data, this program will not run correctly.
           http://www.dragonwins.com/domains/getteched/bmp/bmpfileformat.htm
"""
import sys  # argv
import numpy as np
import nimfa

def read_image(file_path):
    """ Reads the image provided in file_path and returns the data as a
    list containing binary data."""
    image_data = None

    with open(file_path, 'rb') as fil:
        image_data = fil.read()

    return image_data


def get_offset_from_header(image_file):
    """ Returns offset from the header."""
    bt_off = int.from_bytes(image_file[10:14], byteorder='little')

    return bt_off


def get_wid_hei_from_header(image_file):
    """ Returns width and height from the image header."""
    bi_width = int.from_bytes(image_file[18:22], byteorder='little')
    bi_height = int.from_bytes(image_file[22:26], byteorder='little')

    return bi_width, bi_height


def calc_scanline_len(width):
    """ Calculates the actual length of data in one line of the image
    matrix. See bmpfileformat for more information about scanlines."""
    scanline_len = width * 3
    if scanline_len % 4 != 0:
        scanline_len = scanline_len + (4 - scanline_len % 4)

    return scanline_len


def create_new_imagefile(img_matrix, orig_img_file, offset, scanline_len):
    """ Creates a new image file based on the image matrix. Original image
    file is used to obtain the header."""
    new_img = []
    new_img.extend(orig_img_file[:offset])
    img_matrix = img_matrix.tolist()
    null_padding_bytes = scanline_len - len(img_matrix[0])
    null_bytes = []

    for _ in range(null_padding_bytes):
        null_bytes.extend([0])

    for row in img_matrix:
        new_img.extend(row)
        new_img.extend(null_bytes)

    with open("./new_image.bmp", "wb") as fil:
        fil.write(bytes(new_img))

    print("Image file created as ./new_image.bmp")


def perform_nmf(img_matrix, max_iter, rank):
    """ Performs NMF on the provided image matrix. Returned matrix is of
    type np.uint32!"""
    print("Running NMF. Params: max_iter:", max_iter, "rank:", rank)
    V = img_matrix
    nmf = nimfa.Nmf(V, max_iter=max_iter, rank=rank, update='euclidean', objective='fro')
    nmf_fit = nmf()
    W = nmf_fit.basis()  # np.float64
    H = nmf_fit.coef()  # np.float64
    
    mult_matrix = W*H
    mult_matrix = np.rint(mult_matrix)  # rounds but keeps np.float64
    mult_matrix = mult_matrix.astype(np.uint32)

    return mult_matrix


def create_8bit_matrix(image_file, scanline_len, width, height, offset):
    """ Creates a numpy matrix of the image bitmap in the most naive
    representation (8 bit values)."""
    img_mat = [[0 for x in range(width)] for y in range(height)]
    for i in range(height):
        img_mat[i] = list(image_file[offset:(offset + width * 3)])
        offset = offset + scanline_len

    return np.matrix(img_mat)


def compress_naive_8bit(image_file, scanline_len, width, height, offset, nmf_rank, iter_count):
    """ Compresses the provided image file using NMF and representing the image
    matrix as a naive 8 bit matrix."""
    matrix = create_8bit_matrix(image_file, scanline_len, width, height, offset)
    matrix_postnmf = perform_nmf(matrix, iter_count, nmf_rank)
    matrix_postnmf = matrix_postnmf.astype(np.uint8)

    create_new_imagefile(matrix_postnmf, image_file, offset, scanline_len)


def compress_image(image_file, scanline_len, width, height, offset, nmf_rank, iter_count):
    """ Main leading function for compression. Chooses the type of compression
    and calls the corresponding function."""
    compress_naive_8bit(image_file, scanline_len, width, height, offset,
                        nmf_rank, iter_count)


def main():
    if len(sys.argv) != 4:
        print("Usage: nmf_image_compression.py imagefile.bmp nmf_rank iter_count")
        quit()

    image_file = read_image(sys.argv[1])
    offset = get_offset_from_header(image_file)
    width, height = get_wid_hei_from_header(image_file)
    scanline_len = calc_scanline_len(width)
    nmf_rank = int(sys.argv[2])
    iter_count = int(sys.argv[3])

    compress_image(image_file, scanline_len, width, height, offset, nmf_rank, iter_count)


if __name__ == "__main__":
    main()
