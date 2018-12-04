# mastersthesis

NMF - Image compression - performance analysis

## Links
 * Image compression using constrained non-negative matrix factorization - https://pdfs.semanticscholar.org/426c/977bfa62f8018382d3ac69cd6d377e6d7379.pdf
 * Compressed nonnegative matrix factorization is fast and accurate - https://arxiv.org/pdf/1505.04650.pdf
 * An Image Compression Scheme in Wireless Multimedia Sensor Networks Based on NMF - http://www.mdpi.com/2078-2489/8/1/26/pdf
 * MATLAB - Image compression based on NMF http://www.advancedsourcecode.com/nmfcompression.asp (understand the code + rewrite unless there's a C-like language implementation)
 * Nimfa - A Python NMF library - multiple algorithms and parametrization http://nimfa.biolab.si/
 * BMP format description - http://www.dragonwins.com/domains/getteched/bmp/bmpfileformat.htm
 * SVD image compression (also dimensinoality reduction) - http://fourier.eng.hmc.edu/e161/lectures/svdcompression.html - very good paper, should inspire a lot
 * Image Compression Methods using Dimension Reduction and Classification through PCA and LDA: A Review - https://www.ijsr.net/archive/v5i5/NOV163957.pdf - doesn't really show much practical things but there's some nice theory to write about ad. compression and data redundancy
 * Dimensinoality Reduction for Matrix- and Tensor-Coded Data [Part 1] - https://www.youtube.com/watch?v=hmmnRF66hOA - a few slides have very solid images to describe why NMF is different than PCA/what the difference is
 * The Why and How of Nonnegative Matrix Factorization - https://arxiv.org/pdf/1401.5226.pdf - absolutely excellent paper showing usecases of NMF.
 * Online SVD compression demo (SVD explanation + demo for own images) - http://timbaumann.info/svd-image-compression-demo/
 * SVD Image compression slides (quick summarising presentation) - https://www.slideshare.net/AishwaryaKM1/singular-value-decomposition-image-compression
 * NMF decomposition scikit-learn https://scikit-learn.org/stable/modules/decomposition.html#non-negative-matrix-factorization-nmf-or-nnmf
 * NMF x PCA relationship - http://www.columbia.edu/~jwp2128/Teaching/E4903/papers/nmf_nature.pdf

## Topics to research
 * NMF implementation library - OK
 * What sorts of NMF algos are there? Is there one that's clearly superior for this purpose? - ONGOING
 * NMF rank v\*r and r\*w - any heuristic for choosing r? what sort of effect does it have on the images? - ONGOING
 * RGB/CMYK/other? - any difference on the performance?
 * NMF producing floating/decimal numbers - can be larger than the original data - how much of a problem can this be? can the floats be cut/rounded to a certain value or represented in a whole different way? how would that hurt the compression? - ONGOING
 * Followup to previous - https://docs.python.org/3.7/library/decimal.html - how much can decimals be controlled in python/how well can be custom numeric types be implemented in python? or should I just use c/c++? - OK
 * Is it necessary to restrict the factorization only to non-negative matrices? - ONGOING
 * Performance metrics - compression ratio, peak signal-to-noise ratio
 * Constrained NMF - papers seem to imply that CNMF is the way to go - Which constraints can even be used? Not sparsity for sure.

## New stuff/ideas
 * NMF clearly very lossy when considering the numbers as 8bits (already "mock" implemented)
 * Idea - NMF is most likely not very accurate when it comes to lower digits - how about the numbers go from 8bits to large numbers?
 * Different representation - loading into 32bit numbers - either the whole row (better compression but likely one channel will suffer severely) (legal operation - the row is by default zero padded so that it can be used that way) or per pixel (8 zeros at the end)

## Notes
 * Naive 32-bit matrix - VERY lossy, the code needs to be refactored slightly but clearly a lot of information is being lost - not only are pixels overlapping (rgbr gbrg brgb...) but it seems that NMF might be very inaccurate with BIG numbers.
 * No overlap 32-bit matrix - (rgb0 rgb0 rgb0...) yet to be implemented
 * No overlap 24-bit matrix (rgb rgb rgb...) - should be possible to calc with with numpy, research how
