import numpy as np
from scipy.linalg import (blas, lapack)

def ab01nd(jobz, n, m, a, lda, b, ldb, tol=0.0):
    zero = np.float64(0.0)
    one = np.float64(1.0)

    # Test the input scalar arguments.
    if jobz not in ['N', 'F', 'I']:
        raise ValueError("JOBZ must be 'N', 'F', or 'I'")
    if n < 0:
        raise ValueError("N must be non-negative")
    if m < 0:
        raise ValueError("M must be non-negative")
    if lda < max(1, n):
        raise ValueError("LDA must be at least max(1, N)")
    if ldb < max(1, n):
        raise ValueError("LDB must be at least max(1, N)")

    ncont = 0
    indcon = 0
    ljobf = jobz in ['F']
    ljobi = jobz in ['I']
    ljobz = ljobf or ljobi

    # Quick return if possible.
    if min(n, m) == 0:
        if n > 0:
            z = np.zeros((n, n))
            tau = np.zeros(n)
            if ljobi:
                np.fill_diagonal(z, 1.0)
            dwork = np.array([one], dtype=np.float64)
        else:
            z = np.empty((0,0), dtype=np.float64)
            tau = np.empty(0, dtype=np.float64)
            dwork = np.array([one], dtype=np.float64)
        return z, tau, dwork

    # Calculate the absolute norms of A and B (used for scaling).
    anorm = blas.dlange('M', n, n, a, lda)
    bnorm = blas.dlange('M', n, m, b, ldb)

    # Return if matrix B is zero.
    if bnorm == zero:
        z = np.zeros((n, n))
        tau = np.zeros(n)
        if ljobi:
            np.fill_diagonal(z, 1.0)
        dwork = np.array([one], dtype=np.float64)
        return z, tau, dwork

    # Scale (if needed) the matrices A and B.
    nblk = np.zeros(n, dtype=np.int32)
    info = np.zeros(1, dtype=np.int32)
    lapack.mb01pd('Scale', 'G', n, n, 0, 0, anorm, 0, nblk, a, lda, info)
    lapack.mb01pd('Scale', 'G', n, m, 0, 0, bnorm, 0, nblk, b, ldb, info)

    # Compute the Frobenius norm of [B A] (used for rank estimation).
    fnrm = blas.dlange('F', n, m, b, ldb)

    toldef = tol
    if toldef <= zero:
        toldef = np.float64(n*n) * lapack.dlamch('Precision')

    wrkopt = np.int32(1)
    ni = np.int32(0)
    itau = np.int32(0)
    ncrt = np.int32(n)
    mcrt = np.int32(m)
    iqr = np.int32(0)

    while True:
        # Rank-revealing QR decomposition with column pivoting.
