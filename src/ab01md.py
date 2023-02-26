import numpy as np
from scipy.linalg import norm, qr, hessenberg

def AB01MD(JOBZ, A, B, TOL=0):
    # Extract dimensions
    N = A.shape[0]
    
    # Initialize outputs
    Z = np.eye(N)
    TAU = np.zeros(N)
    NCONT = 0
    
    # Check JOBZ argument
    LJOBF = JOBZ == 'F'
    LJOBI = JOBZ == 'I'
    LJOBZ = LJOBF or LJOBI
    
    # Calculate the absolute norms of A and B (used for scaling)
    ANORM = norm(A, ord='fro')
    BNORM = norm(B, ord=1)
    
    # Return if matrix B is zero
    if BNORM == 0:
        if LJOBF:
            return np.zeros((N, N)), np.zeros(N)
        elif LJOBI:
            return np.eye(N)
        else:
            return
    
    # Scale (if needed) the matrices A and B
    ANORM_SCALING, _ = hessenberg(A)
    BNORM_SCALING, _ = hessenberg(B.reshape(-1, 1))
    ANORM_SCALING = norm(ANORM_SCALING, ord='fro')
    BNORM_SCALING = norm(BNORM_SCALING, ord=1)
    A = A / ANORM_SCALING
    B = B / BNORM_SCALING
    
    # Calculate the Frobenius norm of A and the 1-norm of B (used for controllability test)
    FANORM = norm(A, ord='fro')
    FBNORM = norm(B, ord=1)
    
    # Set default tolerance if TOL <= 0
    if TOL <= 0:
        THRESH = float(N) * np.finfo(float).eps
        TOL = THRESH * max(FANORM, FBNORM)
    
    ITAU = 0
    if FBNORM > TOL:
        # B is not negligible compared with A
        if N > 1:
            # Transform B by a Householder matrix Z1
            H, B1 = np.zeros(N), np.zeros(N)
            H[1:] = B[1:]
            B1[0] = B[0]
            H[0], tau = qr(H[:, None], mode='economic')
            B = np.zeros(N)
            B[0] = 1.0
            A = A - 2 * H @ (H.T @ A)
            A = A - 2 * A @ H @ H.T
            B[1:] = H[:, 0]
            TAU[0] = tau
            ITAU += 1
        else:
            B1 = B.copy()
        
        # Reduce modified A to upper Hessenberg form
        # by an orthogonal similarity transformation with matrix Z2
        A, TAU[ITAU:], _ = hessenberg(A, calc_q=True)
        
        # Accumulate the orthogonal transformations used
        if LJOBZ:
            if N > 1:
                Q, R = qr(B[1:, None])
                Z[1:, :] = Z[1:, :] @ Q
            if N > 2:
                Z[2:, 1:] = Z[2:, 1:] - np.outer(TAU[1:], Z[2:, 0]) @ Z[1, :]

