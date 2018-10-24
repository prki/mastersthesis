""" Notes: the script does not check the validity of the file/image header.
          if the file is corrupt OR the header tries to read more than the
          image data, this program will not run correctly.
           http://www.dragonwins.com/domains/getteched/bmp/bmpfileformat.htm
       """
import sys  # argv


def read_image(file_path):
    """ Reads the image provided in filePath and returns the data as a binary
       array."""
    image_data = None

    with open(file_path, 'rb') as fil:
        image_data = fil.read()

    return image_data


def parse_file_header(image_file):
    """ @return offset - the only relevant thing from the header for us. """
    bf_type = image_file[:2]
    bf_size = int.from_bytes(image_file[2:6], byteorder='little')
    bf_off = int.from_bytes(image_file[10:14], byteorder='little')

    print(bf_type, bf_size, bf_off)

    return bf_off


def parse_image_header(image_file):
    """ Only parses width/height as the rest is irrelevant."""
    bi_width = int.from_bytes(image_file[18:22], byteorder='little')
    bi_height = int.from_bytes(image_file[22:26], byteorder='little')

    print("width:", bi_width, "height:", bi_height)

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
    null_padding_bytes = scanline_len - len(img_matrix[0])
    null_bytes = []
    for i in range(null_padding_bytes):
        null_bytes.extend([0])

    for i in range(len(img_matrix)):
        new_img.extend(img_matrix[i])
        new_img.extend(null_bytes)

    with open("./new_image.bmp", "wb") as fil:
        fil.write(bytes(new_img))


def compress_image(image_file, scanline_len, width, height, offset):
    """ Compresses the image using NMF.
       Parses the image data into a matrix without the null padding (in order
       to compress less data). TODO what happens next."""
    img_matrix = [[0 for x in range(width)] for y in range(height)]
    orig_offset = offset  # Used in the test function, can be parsed though

    for i in range(height):
        # Important to read only the part without the null padding
        img_matrix[i] = image_file[offset:(offset + width * 3)]
        offset = offset + scanline_len

    create_new_image_file(img_matrix, image_file, orig_offset, scanline_len)


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
