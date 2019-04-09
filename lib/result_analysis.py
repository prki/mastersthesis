import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    methods = ['ycbcr']
    files = ['out-of-focus.png']
    data = pd.read_csv('./results/img_results.csv', delimiter=',')
    data = data.drop_duplicates(subset=('filename', 'nmf_method', 'nmf_rank', 'nmf_iter', 'nmf_seed'))
    print(data)
    #print(data[(data.filename == 'out-of-focus.png') & (data.nmf_method == 'ycbcr')])  # noqa


if __name__ == '__main__':
    main()
