"""
  A script serving as a playground for loading 8bit numbers into 32bit ones in 2 possible ways (so far).
    - 3 8bit numbers + 8 bits of zero padding
    - load the entire buffer of 8bit numbers as 32bits
"""
import numpy as np


def generate_random_array():
    """ Generates an array with numbers in range of [0;255]"""
    return np.random.randint(0, high=255, dtype=np.uint8, size=100)


def generate_random_matrix():
    """ Generates a matrix with numbers in range of [0;255]"""
    #return np.random.randint(0, high=255, dtype=np.uint8, size=(32, 16))
    return np.full((32, 16), 255, dtype=np.uint8)


def load_32bit_nopad(data):
    return np.frombuffer(data, dtype=np.int32)

def unpack_32bit(data):
    return np.frombuffer(data, dtype=np.uint8)

def convert_8to32bit_matrix(mat):
    return mat.view(np.uint32)
 
def convert_32to8bit_matrix(mat):
    return mat.view(np.uint8)

def main():
    #arr = generate_random_array()
    #print(arr)
    #print(unpack_32bit(load_32bit_nopad(arr)))
    mat = generate_random_matrix()
    print(mat)
    mat32 = convert_8to32bit_matrix(mat)
    #mat32 = np.matrix(mat, dtype=np.int32)
    print(mat32)
    mat8 = convert_32to8bit_matrix(mat32)
    print(mat8)


if __name__ == "__main__":
    main()
