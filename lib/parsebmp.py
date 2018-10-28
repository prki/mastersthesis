""" Notes: the script does not check the validity of the file/image header.
          if the file is corrupt OR the header tries to read more than the
          image data, this program will not run correctly.
           http://www.dragonwins.com/domains/getteched/bmp/bmpfileformat.htm
       """
import sys  # argv
import numpy as np
import nimfa


def read_image(file_path):
    """ Reads the image provided in filePath and returns the data as a binary
       array."""
    image_data = None

    with open(file_path, 'rb') as fil:
        image_data = fil.read()

    return image_data


def parse_file_header(image_file):
    """ @return offset - the only relevant thing from the header for us. """
    #bf_type = image_file[:2]
    #bf_size = int.from_bytes(image_file[2:6], byteorder='little')
    bf_off = int.from_bytes(image_file[10:14], byteorder='little')

    return bf_off


def parse_image_header(image_file):
    """ Only parses width/height as the rest is irrelevant."""
    bi_width = int.from_bytes(image_file[18:22], byteorder='little')
    bi_height = int.from_bytes(image_file[22:26], byteorder='little')

    return bi_width, bi_height


def calc_scanline_len(width):
    """ Calculates the actual length of data provided. Used to check that the
       image file is not corrupt."""
    scanline_len = width * 3
    if scanline_len % 4 != 0:
        scanline_len = scanline_len + (4 - scanline_len % 4)

    return scanline_len


def create_new_image_file(img_matrix, orig_img_file, offset, scanline_len):
    """ Creates a new image file based on the image matrix. Original image
       file is used to obtain the header. Currently used as a test function
       but can be used during the decompressing stage. - however with certain
       recalculations (e.g. scanline_len)"""
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


def perform_nmf(img_matrix, max_iter, rank):
    """ Performs the NMF on the image matrix."""
    V = np.matrix(img_matrix, dtype=np.uint8)
    print("Running NMF. Params: max_iter:", max_iter, "rank:", rank)
    nmf = nimfa.Nmf(V, max_iter=max_iter, rank=rank, update='euclidean', objective='fro')
    nmf_fit = nmf()
    W = nmf_fit.basis()  # np.float64
    H = nmf_fit.coef()  # np.float64
    mult_matrix = W*H
    mult_matrix = np.rint(mult_matrix)  # rounds but keeps np.float64!
    mult_matrix = mult_matrix.astype(np.uint8)

    return mult_matrix


def create_8bit_matrix(image_file, scanline_len, width, height, offset):
    """ Creates a numpy matrix of the bitmap in its most basic fashion -
       8bit numbers."""
    print("Creating an 8bit matrix.")
    img_mat = [[0 for x in range(width)] for y in range(height)]
    for i in range(height):
        img_mat[i] = list(image_file[offset:(offset + width * 3)])
        offset = offset + scanline_len

    return np.matrix(img_mat)


def convert_8to32bit_matrix(mat_8bit):
    """ Converts the numpy uint8 matrix into one where following elements get
    packed into uint8s."""
    print("Converting 8bit matrix to a 32bit one.")
    ret_mat = np.zeros(shape=(mat_8bit.shape[0], int(mat_8bit.shape[1]/4)))
    for i, row in enumerate(mat_8bit):
        ret_mat[i] = np.frombuffer(row, dtype=np.uint32)

    return ret_mat


def compress_image(image_file, scanline_len, width, height, offset):
    """ Compresses the image using NMF.
       Parses the image data into a matrix without the null padding (in order
       to compress less data). NMF compression can be done with various
       parameters."""
    matrix_8bit = create_8bit_matrix(image_file, scanline_len, width, height, offset)
    matrix_32bit = convert_8to32bit_matrix(matrix_8bit)

    img_matrix_postnmf = perform_nmf(matrix_32bit, 300, 50)

    create_new_image_file(img_matrix_postnmf, image_file, offset, scanline_len)


def main():
    if len(sys.argv) == 1:
        print("Usage: parsebmp.py file")
        quit()

    image_file = read_image(sys.argv[1])
    offset = parse_file_header(image_file)
    width, height = parse_image_header(image_file)
    scanline_len = calc_scanline_len(width)

    compress_image(image_file, scanline_len, width, height, offset)


if __name__ == "__main__":
    main()
