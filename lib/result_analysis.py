import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    methods = ['ycbcr']
    files = ['out-of-focus.png']
    data = pd.read_csv('results.csv', delimiter=',', skiprows=1)
    print(data[(data.filename == 'out-of-focus.png') & (data.nmf_method == 'ycbcr')])  # noqa

    for method in methods:
        for fil in files:



if __name__ == '__main__':
    main()
