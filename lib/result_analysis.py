import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_ssim_graph(data, method):
    data = data[(data.nmf_method == method)]
    filenames = data['filename'].drop_duplicates().values.tolist()
    filenames.sort()
    for filename in filenames:
        df = data[['filename', 'nmf_iter', 'ssim']]
        df = df.loc[df['filename'] == filename]
        np_ranks = np.array(df['nmf_iter'].drop_duplicates().values.tolist())
        np_ssims = np.array(df['ssim'].drop_duplicates().values.tolist())
        plt.subplot(2, 1, 1)
        plt.plot(np_ranks, np_ssims, label=filename[:-4])  # -4 - remove the .png
        #plt.legend([filename])
    plt.legend()
    plt.title('SSIM and PSNR, variable max iterations, YCbCr scheme.')
    plt.xlabel('Maximum number iterations')
    plt.ylabel('SSIM')
    #plt.show()
    #plt.savefig('ssim_ycbcr.pdf')


def create_psnr_graph(data, method):
    data = data[(data.nmf_method == method)]
    filenames = data['filename'].drop_duplicates().values.tolist()
    filenames.sort()
    for filename in filenames:
        df = data[['filename', 'nmf_iter', 'psnr']]
        df = df.loc[df['filename'] == filename]
        np_ranks = np.array(df['nmf_iter'].drop_duplicates().values.tolist())
        np_psnrs = np.array(df['psnr'].values.tolist())
        #print(np_ranks)
        #print(np_psnrs)
        plt.subplot(2, 1, 2)
        plt.plot(np_ranks, np_psnrs, label=filename[:-4])

    plt.legend()
    #plt.title('PSNR of benchmark images, variable rank. YCbCr scheme.')
    plt.xlabel('Maximum number of iterations')
    plt.ylabel('PSNR [dB]')
    #plt.show()
    #plt.savefig('psnr_ycbcr.pdf')


def create_time_graph(data, method):
    data = data[(data.nmf_method == method)]
    filenames = data['filename'].drop_duplicates().values.tolist()
    filenames.sort()
    for filename in filenames:
        df = data[['filename', 'compr_time', 'nmf_iter']]
        df = df.loc[df['filename'] == filename]
        np_ranks = np.array(df['nmf_iter'].drop_duplicates().values.tolist())
        np_times = np.array(df['compr_time'].drop_duplicates().values.tolist())

        #plt.subplot(2, 1, 1)
        plt.plot(np_ranks, np_times, label=filename[:-4])

    plt.title('Compr. time of benchmark images, variable max iters, YCbCr scheme')
    plt.xlabel('Maximum number of iterations')
    plt.ylabel('Time [s]')
    plt.legend()


def create_ratio_graph(data, method):
    data = data[(data.nmf_method == method)]
    filenames = data['filename'].drop_duplicates().values.tolist()
    filenames.sort()
    for filename in filenames:
        df = data[['filename', 'compr_ratio', 'nmf_rank']]
        df = df.loc[df['filename'] == filename]
        np_ranks = np.array(df['nmf_rank'].drop_duplicates().values.tolist())
        np_ratios = np.array(df['compr_ratio'].drop_duplicates().values.tolist())

        plt.subplot(2, 1, 2)
        plt.plot(np_ranks, np_ratios, label=filename[:-4])

    plt.xlabel('Rank')
    plt.ylabel('Compression ratio')
    plt.legend()

def compare_jpeg(data, method):
    data = data[(data.nmf_method == method)]
    filenames = data['filename'].drop_duplicates().values.tolist()
    filenames.sort()

    for filename in filenames:
        df = data.loc[data['filename'] == filename]
        max_psnr = max(df['psnr'])
        max_ssim = max(df['ssim'])
        print('file:', filename, 'max_psnr:', max_psnr)
        print('file:', filename, 'max_ssim:', max_ssim)
        print('file:', filename, 'psnr_jpeg:', set(df['psnr_jpeg']))
        print('file:', filename, 'ssim_jpeg:', set(df['ssim_jpeg']))


def barchart_imgsizes(data):
    method = 'ycbcr'
    filenames = data['filename'].drop_duplicates().values.tolist()
    filenames.sort()
    n_bars = len(filenames)
    fig, ax = plt.subplots()
    bar_width = 0.2
    index = np.arange(n_bars)
    jpg_sizes = []
    nmf_sizes = []
    orig_sizes = []


    for filename in filenames:
        r = 20
        df = data.loc[(data['filename'] == filename) & (data['nmf_rank'] == r) & (data['nmf_method'] == method)]
        orig_size = df['orig_size'].values[0]
        jpg_size = df['jpeg_size'].values[0]
        nmf_size = df['nmf_size'].values[0]
        orig_sizes.append(orig_size)
        nmf_sizes.append(nmf_size)
        jpg_sizes.append(jpg_size)

    plt.title('Image size comparison')
    rects1 = ax.bar(index, orig_sizes, bar_width, label='Orig. size')
    rects2 = ax.bar(index + bar_width, nmf_sizes, bar_width, label='NMF size (rank 20)')
    rects3 = ax.bar(index + bar_width + bar_width, jpg_sizes, bar_width, label='JPEG size')
    ax.legend()
    ax.set_xticks(index + bar_width/2)
    ax.set_ylabel('Size [B]')
    ax.set_xlabel('Image name')
    filenames_cut = []
    for filename in filenames:
        filenames_cut.append(filename[:-4])  # cut extension
    ax.set_xticklabels(filenames_cut, rotation=10)

    fig.tight_layout()

    plt.savefig('filesize_comparison.pdf')


def main():
    #data = pd.read_csv('./results/img_results_varrank.csv', delimiter=',')
    data = pd.read_csv('./results/img_results.csv', delimiter=',')
    data = data.drop_duplicates(subset=('filename', 'nmf_method', 'nmf_rank', 'nmf_iter', 'nmf_seed'))
    #print(data[(data.filename == 'out-of-focus.png') & (data.nmf_method == 'ycbcr')])  # noqa
    #create_ssim_graph(data, 'ycbcr')
    #create_psnr_graph(data, 'naive24')
    #create_ssim_graph(data, 'ycbcr')
    #create_psnr_graph(data, 'ycbcr')
    #plt.tight_layout()
    #plt.show()
    #plt.savefig('psnr_ssim_maxiter_ycbcr.pdf')
    #plt.tight_layout()
    #plt.savefig('psnr_ssim_ycbcr.pdf')
    #create_psnr_graph(data, 'naive24')
    #create_psnr_graph(data, 'naive24')
    #compare_jpeg(data, 'ycbcr')
    #barchart_imgsizes(data)
    create_time_graph(data, 'ycbcr')
    plt.tight_layout()
    plt.savefig('comprtime_maxiter_ycbcr.pdf')
    #create_ratio_graph(data, 'ycbcr')
    #plt.tight_layout()
    #plt.savefig('comprtimeratio_ycbcr.pdf')


if __name__ == '__main__':
    main()
