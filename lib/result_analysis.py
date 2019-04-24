import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_ssim_graph(data, method):
    data = data[(data.nmf_method == method)]
    filenames = data['filename'].drop_duplicates().values.tolist()
    filenames.sort()
    for filename in filenames:
        df = data[['filename', 'nmf_rank', 'ssim']]
        df = df.loc[df['filename'] == filename]
        np_ranks = np.array(df['nmf_rank'].drop_duplicates().values.tolist())
        np_ssims = np.array(df['ssim'].drop_duplicates().values.tolist())
        plt.plot(np_ranks, np_ssims, label=filename[:-4])  # -4 - remove the .png
        #plt.legend([filename])
    plt.legend()
    plt.title('SSIM of benchmark images with variable rank.')
    plt.xlabel('Rank')
    plt.ylabel('SSIM')
    plt.show()


def create_psnr_graph(data, method):
    data = data[(data.nmf_method == method)]
    filenames = data['filename'].drop_duplicates().values.tolist()
    filenames.sort()
    for filename in filenames:
        df = data[['filename', 'nmf_rank', 'psnr']]
        df = df.loc[df['filename'] == filename]
        np_ranks = np.array(df['nmf_rank'].drop_duplicates().values.tolist())
        np_psnrs = np.array(df['psnr'].values.tolist())
        print(df)
        #print(np_ranks)
        #print(np_psnrs)
        plt.plot(np_ranks, np_psnrs, label=filename[:-4])

    plt.legend()
    plt.title('PSNR of benchmark images with variable rank.')
    plt.xlabel('Rank')
    plt.ylabel('PSNR [dB]')
    plt.show()


def main():
    data = pd.read_csv('./results/img_results.csv', delimiter=',')
    data = data.drop_duplicates(subset=('filename', 'nmf_method', 'nmf_rank', 'nmf_iter', 'nmf_seed'))
    #print(data[(data.filename == 'out-of-focus.png') & (data.nmf_method == 'ycbcr')])  # noqa
    #create_ssim_graph(data, 'ycbcr')
    create_psnr_graph(data, 'ycbcr')
    #create_psnr_graph(data, 'naive24')
    #create_psnr_graph(data, 'naive24')


if __name__ == '__main__':
    main()
