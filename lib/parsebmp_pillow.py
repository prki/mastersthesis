import sys
import numpy as np
import nimfa
import scipy.fftpack
#import scipy.fftpack.dct as dct
#import scipy.fftpack.idct as idct
from PIL import Image

def bmp2grayscale(img):
    """ Converts a PIL Image to a grayscale PIL Image. Returns a
    transpoed numpy matrix.
    """
    img_new = img.convert('L')
    npim = np.array(img_new)

    return np.transpose(npim)


def perform_nmf(np_matrix, max_iter, rank):
    """ Performs NMF on the image matrix given as a numpy matrix."""
    print('Running NMF. Params: max_iter:', max_iter, 'rank:', rank)
    V = np_matrix
    nmf = nimfa.Nmf(V, max_iter=max_iter, rank=rank, update='euclidean', objective='fro')
    #nmf = nimfa.Lsnmf(V, seed='random_vcol', rank=rank, max_iter=max_iter)
    nmf_fit = nmf()
    W = nmf_fit.basis()  # np.float64
    H = nmf_fit.coef()  # np.float64
    mult_matrix = W*H
    #mult_matrix = np.rint(mult_matrix)  # rounds but keeps np.float64!
    #mult_matrix = mult_matrix.astype(np.uint8)

    return mult_matrix


def prk_compression_scheme(np_data):
    """ Compression scheme which performs NMF on 8x8 square matrices."""
    submats = []
    submat_size = 32

    for i in range(0, np_data.shape[0], submat_size):
        for j in range(0, np_data.shape[1], submat_size):
            submat = np_data[i:i+submat_size, j:j+submat_size]
            submats.append(submat)

    submats_afternmf = []

    print(len(submats))

    for submat in submats:
        submat_after = perform_nmf(submat, 50, 8)
        submats_afternmf.append(submat_after)

    #A =  np.array(submats_afternmf)
    #m,n,r = A.shape
    #out = A.reshape(-1,2,n,r).transpose(0,2,1,3).reshape(-1,2*r)

    #print(out.shape)
    #arr2im = Image.fromarray(np.transpose(out))


def create_singlecolor_matrix(np_img, idx):
    """idx - dimension/color of each pixel component in rgb
    (0 - red, 1 - green, 2 - blue)"""
    w = np_img.shape[0]
    h = np_img.shape[1]

    mat = np.zeros((w, h), dtype=np.uint8)

    for i in range(w):
        for j in range(h):
            mat[i][j] += np_img[i][j][idx]

    return mat


def parse_3d_matrix(np_img):
    """ Parses a 3D RGB Matrix of shape (w, h, 3), where each [w][h] position
    has a tuple containing (r, g, b) elements. Returns a tuple (rmatrix, gmatrix,
    bmatrix), where each matrix includes only the respective color component of
    each pixel.
    """
    w = np_img.shape[0]
    h = np_img.shape[1]

    rmat = create_singlecolor_matrix(np_img, 0)
    gmat = create_singlecolor_matrix(np_img, 1)
    bmat = create_singlecolor_matrix(np_img, 2)

    return rmat, gmat, bmat


def reconstruct_3d_matrix(rmat, gmat, bmat):
    """ Reconstructs the RGB image saved in the 3 matrices into the matrix
    numpy/PIL works with - in shape (width, height, 3)."""
    w = rmat.shape[0]
    h = rmat.shape[1]

    mat = np.zeros((w, h, 3), dtype=np.uint8)

    for i in range(w):
        for j in range(h):
            mat[i][j][0] += rmat[i,j]#rmat[i][j]
            mat[i][j][1] += bmat[i,j]#bmat[i][j]
            mat[i][j][2] += gmat[i,j]#gmat[i][j]

    return mat


def create_8bit_rgb_matrix(rmat, gmat, bmat):
    """ Creates a 2D RGB matrix in the form of
    (r g b r g b r g b ...
     r g b r g b r g b ...
     r g b r g b r g b ...)
    Attrs: Matrices of R/G/B components."""
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


def decompose_8bit_rgb_matrix(mat):
    """ Decomposes a 2D RGB Matrix in the form of
    (r g b r g b r g b ...
     r g b r g b r g b ...
     r g b r g b r g b ...)
     into 3 matrices containing each component."""
    w = mat.shape[0]
    h = int(mat.shape[1] / 3)
    
    rmat = np.zeros((w, h), dtype=np.uint8)
    gmat = np.zeros((w, h), dtype=np.uint8)
    bmat = np.zeros((w, h), dtype=np.uint8)

    col = 0
    for i in range(w):
        for j in range(0, mat.shape[1], 3):
            rmat[i][col] += mat[i, j]#mat[i][j]
            bmat[i][col] += mat[i, j+1]#mat[i][j+1]
            gmat[i][col] += mat[i, j+2]#mat[i][j+2]
            col += 1
        col = 0

    return rmat, bmat, gmat


def experiment_dct(np_gray):
    print("Attempting DCT:")
    dct_img = scipy.fftpack.dct(np_gray, norm='ortho')
    min_val = np.amin(dct_img)
    dct_img += abs(min_val)  # Make non-negative

    nmf_dct = perform_nmf(dct_img, 200, 300)
    nmf_dct -= abs(min_val)

    nmf_dct_inv = scipy.fftpack.idct(nmf_dct, norm='ortho')
    nmf_dct_inv = np.rint(nmf_dct_inv)
    nmf_dct_inv = nmf_dct_inv.astype(np.uint8)
    arr2im = Image.fromarray(np.transpose(nmf_dct_inv))
    arr2im.show()
    print("Without DCT:")
    nmf_gray = perform_nmf(np_gray, 200, 300)
    nmf_gray = np.rint(nmf_gray)
    nmf_gray = nmf_gray.astype(np.uint8)
    #arr2im = Image.fromarray(np.transpose(nppixels_t_grayscale))
    #arr2im.show()
    arr2im = Image.fromarray(np.transpose(nmf_gray))
    arr2im.show()
    
    #dct_img_inv = scipy.fftpack.idct(dct_img, norm='ortho')
    #print(dct_img_inv.shape)
    #print(dct_img_inv)



def main():
    if len(sys.argv) == 1:
        print('Usage: parsebmp.py <filepath_to_bmp>')
        raise SystemExit(1)

    img = Image.open(sys.argv[1])
    nppixels = np.array(img)
    nppixels_t = np.transpose(nppixels)

    #print(nppixels_t)

    experiment_dct(bmp2grayscale(img))

    # RGB image
    """
    npt = np.transpose(nppixels, (1, 0, 2))
    rmat, gmat, bmat = parse_3d_matrix(npt)  # decompose big matrix
    rmat_nmf = perform_nmf(rmat, 200, 100)
    gmat_nmf = perform_nmf(gmat, 200, 100)
    bmat_nmf = perform_nmf(bmat, 200, 100)
    mat_nmf = reconstruct_3d_matrix(rmat_nmf, gmat_nmf, bmat_nmf)
    mat_nmf = np.transpose(mat_nmf, (1, 0, 2))
    arr2im = Image.fromarray(mat_nmf)
    arr2im.show()
    """
    #mat = create_8bit_rgb_matrix(rmat, gmat, bmat)  # create 2D 8bit repr.
    #mat_nmf = perform_nmf(mat, 800, 100)
    #rmat, gmat, bmat = decompose_8bit_rgb_matrix(mat_nmf)
    #mat = reconstruct_3d_matrix(rmat, gmat, bmat)
    #mat = np.transpose(mat, (1, 0, 2))

    #arr2im = Image.fromarray(mat)
    #arr2im.show()

    #rmat, gmat, bmat = decompose_8bit_rgb_matrix(mat)  # decompose back into each component
    #mat = reconstruct_3d_matrix(rmat, gmat, bmat)  # reconstruct from each component
    
    #mat = np.transpose(mat, (1, 0, 2))
    #arr2im = Image.fromarray(mat)
    #arr2im.show()

    # build representing matrices here
    # TODO
    #nppixels_t_8bit = create_8bit_matrix(nppixels_t)
    #nppixels_t_32bit = create_32bit_matrix(nppixels_t)
    #nppixels_t_24bit = create_24bit_matrix(nppixels_t)
    #nppixels_t_grayscale = bmp2grayscale(img)

    #prk_compression_scheme(nppixels_t_grayscale)

    #arr2im = Image.fromarray(np.transpose(nppixels_t_grayscale))
    #arr2im.show()

    #nmf_nppixels_t_grayscale = perform_nmf(nppixels_t_grayscale, 200, 1)  # Simple compression
    #nmf_nppixels_t = perform_nmf(nppixels_t_grayscale, 2000, 100)
    #arr2im = Image.fromarray(np.transpose(nmf_nppixels_t))
    #arr2im.show()

    #arr2im = Image.fromarray(np.transpose(nmf_nppixels_t_grayscale))
    #arr2im.show()
    #print(nppixels_t_grayscale.shape)



if __name__ == "__main__":
    main()
