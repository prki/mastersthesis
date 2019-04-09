""" Script which compresses images using compression schemes based on
non-negative matrix factorization, as described in the thesis.
"""
import csv
import time
import logging
import sys
import zlib
import numpy as np
import nimfa
from PIL import Image
import img_utils


__SIZE_UINT8 = np.uint8(5).itemsize
__SIZE_UINT32 = np.uint32(5).itemsize
__SIZE_FLOAT32 = np.float32(5).itemsize


def resolve_path(filepath):
    if '\\' not in filepath and '/' not in filepath:
        return filepath
    else:
        pos = filepath.rfind('\\')
        if pos == -1:
            pos = filepath.rfind('/')
        return filepath[pos+1:]  # +1 to not count in the slash


def output_to_csv(lst):
    logging.info('Appending results to a csv file.')
    with open('results/img_results.csv', 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(lst)


def reshape_flat_data(buf, w, rank, h, dtype=np.float32):
    itemsize = dtype(5).itemsize  # cant ask dtype explicitly
    wr = w * rank * itemsize
    buf_flat_w = np.frombuffer(buf[:wr], dtype=dtype)
    buf_flat_h = np.frombuffer(buf[wr:], dtype=dtype)

    return buf_flat_w.reshape((w, rank)), buf_flat_h.reshape((rank, h))


def output_test_results(compr_time, decompr_time, img_filepath, compressed_image, decompr_path, decompr_img,
                        comprscheme, nmfrank, nmfiter, nmfseed):
    decompr_img = decompr_img.convert('RGB')
    imgname = resolve_path(img_filepath)
    decompr_img.save(decompr_path)
    orig_size = img_utils.get_orig_size(img_filepath)
    jpeg_size = img_utils.get_jpeg_size(img_filepath)
    compr_ratio = orig_size / len(compressed_image)
    mse, psnr = img_utils.calc_metrics(img_filepath, decompr_path)
    mse_luma, psnr_luma = img_utils.calc_metrics_luma(img_filepath, decompr_path)
    jpeg_mse, jpeg_psnr, jpeg_mse_luma, jpeg_psnr_luma = img_utils.calc_metrics_jpeg(img_filepath)

    csv_out = [imgname, orig_size, jpeg_size, comprscheme, nmfrank, nmfiter, nmfseed,
               len(compressed_image), compr_ratio, compr_time, decompr_time, psnr,
               psnr_luma, mse, mse_luma, jpeg_psnr, jpeg_psnr_luma, jpeg_mse, jpeg_mse_luma]

    output_to_csv(csv_out)


def perform_nmf(np_matrix, max_iter, rank, dtype, seed):
    """ Function performing NMF on a matrix provided as a numpy matrix.
    Function rounds the values and represents them as uint8 after
    having performed NMF.
    """
    logging.info('Running NMF. Max iters: %s, rank: %s', max_iter, rank)
    V = np_matrix
    nmf = nimfa.Nmf(V, seed=seed, max_iter=max_iter, rank=rank, update='euclidean', objective='fro')
    nmf_fit = nmf()
    W = nmf_fit.basis()  # np.float64
    H = nmf_fit.coef()  # np.float64

    return W.astype(dtype), H.astype(dtype)


def compress_naive_nmf(np_w, np_h):
    logging.info('Compressing naive 24bit RGB image data.')
    w_w = np_w.shape[0]
    w_h = np_h.shape[1]
    rank = np_w.shape[1]

    w_flat = np.asarray(np_w.reshape(-1)).flatten()
    h_flat = np.asarray(np_h.reshape(-1)).flatten()

    header = np.array([w_w, w_h, rank], dtype=np.uint32)
    xs = bytearray()
    xs.extend(header)

    concat = np.concatenate([w_flat, h_flat])
    concat_bytes = concat.tobytes()
    xs.extend(concat_bytes)

    data = bytes(xs)
    data_compr = zlib.compress(data, 9)

    return data_compr


def decompress_naive_nmf(compressed_data):
    """[header, w_flat, h_flat]"""
    logging.info('Decompressing naive 24bit RGB image data.')
    data = zlib.decompress(compressed_data)
    header = data[:3*__SIZE_UINT32]  # 3 elements sizeof uint32
    header = np.frombuffer(header, dtype=np.uint32)
    data = data[3*__SIZE_UINT32:]

    w = header[0]
    h = header[1]
    r = header[2]

    w_mat, h_mat = reshape_flat_data(data, w, r, h, dtype=np.float32)

    wh_mat = np.mat(w_mat) * np.mat(h_mat)
    # Round floats and represent matrix as one containing 8bit values
    wh_mat = np.rint(wh_mat).astype(np.uint8)

    rmat_nmf, gmat_nmf, bmat_nmf = img_utils.decompose_24bit_rgb_matrix(wh_mat)
    img_nmf = img_utils.reconstruct_image(rmat_nmf, gmat_nmf, bmat_nmf)
    arr2im = Image.fromarray(img_nmf)

    return arr2im


def naive_rgb_scheme(img_filepath, max_iter, rank, dtype, seed):
    logging.info('Running naive RGB compression scheme.')
    img = Image.open(img_filepath)
    nppixels = np.array(img)
    rmat, gmat, bmat = img_utils.decompose_into_rgb(nppixels)
    mat = img_utils.construct_24bit_rgb_matrix(rmat, gmat, bmat)

    np_w, np_h = perform_nmf(mat, max_iter, rank, dtype, seed)

    compressed_image = compress_naive_nmf(np_w, np_h)

    return compressed_image


def compress_separate_nmf(rmat_w, rmat_h, gmat_w, gmat_h, bmat_w, bmat_h):
    logging.info('Compressing separate RGB NMF image data.')
    w = rmat_w.shape[0]
    h = rmat_h.shape[1]
    r = rmat_w.shape[1]

    rmat_w_flat = np.asarray(rmat_w.reshape(-1)).flatten()
    rmat_h_flat = np.asarray(rmat_h.reshape(-1)).flatten()
    gmat_w_flat = np.asarray(gmat_w.reshape(-1)).flatten()
    gmat_h_flat = np.asarray(gmat_h.reshape(-1)).flatten()
    bmat_w_flat = np.asarray(bmat_w.reshape(-1)).flatten()
    bmat_h_flat = np.asarray(bmat_h.reshape(-1)).flatten()

    header = np.array([w, h, r], dtype=np.uint32)
    xs = bytearray()
    xs.extend(header)

    concat = np.concatenate([rmat_w_flat, rmat_h_flat, gmat_w_flat, gmat_h_flat,
                             bmat_w_flat, bmat_h_flat])
    concat_bytes = concat.tobytes()
    xs.extend(concat_bytes)

    data = bytes(xs)
    data_compr = zlib.compress(data, 9)

    return data_compr


def decompress_separate_rgb(compressed_data):
    logging.info('Decompressing separate RGB NMF image data.')
    data = zlib.decompress(compressed_data)
    header = data[:3*__SIZE_UINT32]  # 3 elems sizeof uint32
    header = np.frombuffer(header, dtype=np.uint32)
    data = data[3*__SIZE_UINT32:]

    w = header[0]
    h = header[1]
    r = header[2]

    split = int(len(data) / 3)
    rmat_flat = data[:split]
    gmat_flat = data[split:split * 2]
    bmat_flat = data[split * 2:]

    rmat_w, rmat_h = reshape_flat_data(rmat_flat, w, r, h, dtype=np.float32)
    gmat_w, gmat_h = reshape_flat_data(gmat_flat, w, r, h, dtype=np.float32)
    bmat_w, bmat_h = reshape_flat_data(bmat_flat, w, r, h, dtype=np.float32)

    rmat_nmf = np.mat(rmat_w) * np.mat(rmat_h)
    gmat_nmf = np.mat(gmat_w) * np.mat(gmat_h)
    bmat_nmf = np.mat(bmat_w) * np.mat(bmat_h)

    rmat_nmf = np.rint(rmat_nmf).astype(np.uint8)
    gmat_nmf = np.rint(gmat_nmf).astype(np.uint8)
    bmat_nmf = np.rint(bmat_nmf).astype(np.uint8)

    img_nmf = img_utils.reconstruct_image(rmat_nmf, gmat_nmf, bmat_nmf)

    arr2im = Image.fromarray(img_nmf)

    return arr2im


def separate_rgb_scheme(img_filepath, max_iter, rank, dtype, seed):
    logging.info('Running separate RGB compression scheme.')
    img = Image.open(img_filepath)
    nppixels = np.array(img)
    rmat, gmat, bmat = img_utils.decompose_into_rgb(nppixels)
    rmat_w, rmat_h = perform_nmf(rmat, max_iter, rank, dtype, seed)
    gmat_w, gmat_h = perform_nmf(gmat, max_iter, rank, dtype, seed)
    bmat_w, bmat_h = perform_nmf(bmat, max_iter, rank, dtype, seed)

    compressed_image = compress_separate_nmf(rmat_w, rmat_h, gmat_w, gmat_h,
                                             bmat_w, bmat_h)

    return compressed_image


def decompress_ycbcr_nmf(compressed_data):
    """ Compressed data using zlib compression in the following form:
    [header, y_data, cb_w_data, cb_h_data, cr_w_data, cr_h_data]
    Where y_data, cb_data and cr_data are flattened matrices.
    Y_data is represented using uint8, cb_w_data, cb_h_data (or cr_*
    respectively) are matrices represented using float32. Y_data has been
    stored losslessly whereas Cb/Cr information requires to be multiplied, as
    the stored data are the NMF factor matrices.
    """
    logging.info('Decompressing NMF YCbCr image data.')
    data = zlib.decompress(compressed_data)

    header = data[:3*__SIZE_UINT32]  # 3 elements sizeof uint32
    header = np.frombuffer(header, dtype=np.uint32)
    data = data[3*__SIZE_UINT32:]

    cb_w = header[0]
    cb_h = header[1]
    cr_w = cb_w
    cr_h = cb_h
    rank = header[2]

    y_mat_data = data[:cb_h * cb_w * 1]  # sizeof(uint8)
    y_mat_data = np.frombuffer(y_mat_data, dtype=np.uint8)
    data = data[cb_h * cb_w * 1:]  # sizeof(uint8)

    split = int(len(data) / 2)
    cb_flat = data[:split]
    cr_flat = data[split:]

    cb_w_mat, cb_h_mat = reshape_flat_data(cb_flat, cb_w, rank, cb_h)
    cr_w_mat, cr_h_mat = reshape_flat_data(cr_flat, cr_w, rank, cr_h)

    cb_mat = np.mat(cb_w_mat) * np.mat(cb_h_mat)
    cr_mat = np.mat(cr_w_mat) * np.mat(cr_h_mat)

    cb_mat = np.rint(cb_mat)
    cr_mat = np.rint(cr_mat)

    np_y = y_mat_data.reshape((cb_w, cb_h))
    cb_mat = cb_mat.astype(np.uint8)
    cr_mat = cr_mat.astype(np.uint8)

    np_ycbcr_nmf = img_utils.reconstruct_ycbcr_image(np_y, cb_mat, cr_mat)
    arr2im = Image.fromarray(np_ycbcr_nmf, mode='YCbCr')

    return arr2im


def compress_ycbcr_nmf(y_mat, cb_w_mat, cb_h_mat, cr_w_mat, cr_h_mat):
    logging.info('Compressing NMF YCbCr data using zlib.')
    cb_w = cb_w_mat.shape[0]
    cb_h = cb_h_mat.shape[1]
    cr_w = cr_w_mat.shape[0]
    cr_h = cr_h_mat.shape[1]
    rank = cb_w_mat.shape[1]

    cb_w_flat = np.asarray(cb_w_mat.reshape(-1)).flatten()
    cb_h_flat = np.asarray(cb_h_mat.reshape(-1)).flatten()
    cr_w_flat = np.asarray(cr_w_mat.reshape(-1)).flatten()
    cr_h_flat = np.asarray(cr_h_mat.reshape(-1)).flatten()

    header = np.array([cb_w, cb_h, rank], dtype=np.uint32)
    xs = bytearray()
    xs.extend(header)

    y_mat = y_mat.astype(np.uint8)
    y_mat_flat = np.asarray(y_mat.reshape(-1)).flatten()
    xs.extend(y_mat_flat.tobytes())

    concat = np.concatenate([cb_w_flat, cb_h_flat, cr_w_flat, cr_h_flat])
    concat_bytes = concat.tobytes()
    xs.extend(concat_bytes)
    data = bytes(xs)
    concat_compr = zlib.compress(data, 9)

    return concat_compr


def ycbcr_scheme(img_filepath, max_iter, rank, dtype, seed):
    logging.info('Running YCbCr compression scheme.')
    img = Image.open(img_filepath)
    npycbcr = img_utils.pil2ycbcr(img)
    np_y = npycbcr[:, :, 0]
    np_cb = npycbcr[:, :, 1]
    np_cr = npycbcr[:, :, 2]

    np_cb_w, np_cb_h = perform_nmf(np_cb, max_iter, rank, dtype, seed)
    np_cr_w, np_cr_h = perform_nmf(np_cr, max_iter, rank, dtype, seed)

    compressed_image = compress_ycbcr_nmf(np_y, np_cb_w, np_cb_h, np_cr_w, np_cr_h)

    return compressed_image


def test_ycbcr_scheme(img_filepath, max_iter, rank, dtype, seed):
    compr_time_begin = time.time()
    compressed_image = ycbcr_scheme(img_filepath, max_iter, rank, dtype, seed)
    compr_time_end = time.time()
    compr_time = compr_time_end - compr_time_begin

    decompr_time_begin = time.time()
    decompr_img = decompress_ycbcr_nmf(compressed_image)
    decompr_time_end = time.time()
    decompr_time = decompr_time_end - decompr_time_begin

    imgname = resolve_path(img_filepath)
    decompr_path = './imgs/compress/ycbcr/{0}_ycbcr_rank{1}_iter{2}.png'.format(imgname, rank, max_iter)

    output_test_results(compr_time, decompr_time, img_filepath, compressed_image, decompr_path,
                        decompr_img, 'ycbcr', rank, max_iter, seed)


def test_naive_rgb_scheme(img_filepath, max_iter, rank, dtype, seed):
    compr_time_begin = time.time()
    compressed_image = naive_rgb_scheme(img_filepath, max_iter, rank, dtype, seed)
    compr_time_end = time.time()
    compr_time = compr_time_end - compr_time_begin

    decompr_time_begin = time.time()
    decompr_img = decompress_naive_nmf(compressed_image)
    decompr_time_end = time.time()
    decompr_time = decompr_time_end - decompr_time_begin

    imgname = resolve_path(img_filepath)
    decompr_path = './imgs/compress/naive24/{0}_naive24_rank{1}_iter{2}.png'.format(imgname, rank, max_iter)

    output_test_results(compr_time, decompr_time, img_filepath, compressed_image, decompr_path,
                        decompr_img, 'naive24', rank, max_iter, seed)


def test_separate_rgb_scheme(img_filepath, max_iter, rank, dtype, seed):
    compr_time_begin = time.time()
    compressed_image = separate_rgb_scheme(img_filepath, max_iter, rank, dtype, seed)
    compr_time_end = time.time()
    compr_time = compr_time_end - compr_time_begin

    decompr_time_begin = time.time()
    decompr_img = decompress_separate_rgb(compressed_image)
    decompr_time_end = time.time()
    decompr_time = decompr_time_end - decompr_time_begin

    imgname = resolve_path(img_filepath)
    decompr_path = './imgs/compress/separate/{0}_separate_rank{1}_iter{2}.png'.format(imgname, rank, max_iter)

    output_test_results(compr_time, decompr_time, img_filepath, compressed_image, decompr_path,
                        decompr_img, 'separate', rank, max_iter, seed)


def main():
    if len(sys.argv) == 1:
        print('Usage: nmf_img_compressor.py <image_file>')
        raise SystemExit(1)

    img_filepath = sys.argv[1]
    #naive_rgb_scheme(img, 50, 50, dtype=np.float32, seed='nndsvd')
    #separate_rgb_scheme(img, 500, 200)
    #grayscale_scheme(img, 500, 100, dtype=np.float32, seed='random_vcol')
    #test_ycbcr_scheme(img_filepath, 200, 20, dtype=np.float32, seed='nndsvd')
    #ycbcr_scheme(img_filepath, 200, 20, dtype=np.float32, seed='nndsvd')

    #for rank in range(5, 21):
    #    test_ycbcr_scheme(img_filepath, 300, rank, dtype=np.float32, seed='nndsvd')
    #test_naive_rgb_scheme(img_filepath, 50, 50, dtype=np.float32, seed='nndsvd')
    #test_separate_rgb_scheme(img_filepath, 50, 50, dtype=np.float32, seed='nndsvd')
    test_ycbcr_scheme(img_filepath, 200, 20, dtype=np.float32, seed='nndsvd')



if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
