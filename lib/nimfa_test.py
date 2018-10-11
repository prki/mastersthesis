import numpy as np
import nimfa


def main():
    V = np.matrix('1 0 2 4; 0 0 6 3; 4 0 5 6')
    nmf = nimfa.Nmf(V, max_iter=1000, rank=3, update='euclidean', objective='fro')
    nmf_fit = nmf()
    W = nmf_fit.basis()
    H = nmf_fit.coef()
    print("Orig matrix:\n", V)
    print("Basis matrix:\n", W)
    print("Mixture matrix:\n", H)
    multMatrix = W*H
    print("Multiplied matrices:\n", multMatrix)


if __name__ == "__main__":
    main()
