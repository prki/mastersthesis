# mastersthesis

NMF - Image compression - performance analysis

## Links
 * Image compression using constrained non-negative matrix factorization - https://pdfs.semanticscholar.org/426c/977bfa62f8018382d3ac69cd6d377e6d7379.pdf
 * Compressed nonnegative matrix factorization is fast and accurate - https://arxiv.org/pdf/1505.04650.pdf
 * An Image Compression Scheme in Wireless Multimedia Sensor Networks Based on NMF - http://www.mdpi.com/2078-2489/8/1/26/pdf
 * MATLAB - Image compression based on NMF http://www.advancedsourcecode.com/nmfcompression.asp (understand the code + rewrite unless there's a C-like language implementation)
 * Nimfa - A Python NMF library - multiple algorithms and parametrization http://nimfa.biolab.si/

## Topics to research
 * NMF implementation library
 * What sorts of NMF algos are there? Is there one that's clearly superior for this purpose?
 * NMF rank v\*r and r\*w - any heuristic for choosing r? what sort of effect does it have on the images?
 * RGB/CMYK/other? - any difference on the performance?
 * NMF producing floating/decimal numbers - can be larger than the original data - how much of a problem can this be? can the floats be cut/rounded to a certain value or represented in a whole different way? how would that hurt the compression?
 * Is it necessary to restrict the factorization only to non-negative matrices?
 * Performance metrics - compression ratio, peak signal-to-noise ratio
 * Constrained NMF - papers seem to imply that CNMF is the way to go
