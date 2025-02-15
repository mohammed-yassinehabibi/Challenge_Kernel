import numpy as np
from tqdm import tqdm
from itertools import product, combinations
from collections import Counter, defaultdict
import py_stringmatching as sm
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix, csr_matrix
from concurrent.futures import ThreadPoolExecutor
import torch
from scipy.sparse import lil_matrix

def normalize(K):
    D = np.diag(1/np.sqrt(np.diag(K)))
    return np.dot(np.dot(D, K), D)

################################
###     Kernel Functions     ###
################################

def dist_kernel(X_left, X_right):
    """
    Compute the distance kernel matrix between two sets of vectors.

    Parameters
    ----------
    X_left : ndarray
        A 2D array of shape (n_samples_left, n_features).
    X_right : ndarray
        A 2D array of shape (n_samples_right, n_features).

    Returns
    -------
    K_dist : ndarray
        A 2D array of shape (n_samples_left, n_samples_right) containing the pairwise distances.
    """
    K_dist = np.linalg.norm(X_left[:, np.newaxis] - X_right, axis=2)
    return K_dist

def exp_kernel(X_left, X_right, sigma=1):
    """
    Compute the exponential kernel matrix between two sets of vectors.

    Parameters
    ----------
    X_left : ndarray
        A 2D array of shape (n_samples_left, n_features).
    X_right : ndarray
        A 2D array of shape (n_samples_right, n_features).
    sigma : float, optional
        The bandwidth parameter for the exponential kernel (default is 1).

    Returns
    -------
    K_exp : ndarray
        A 2D array of shape (n_samples_left, n_samples_right) containing the exponential kernel values.
    """
    K_dist = dist_kernel(X_left, X_right)
    return normalize(np.exp(-K_dist**2/(2*sigma**2)))

def gaussian_kernel(K, sigma):
    # Compute the diagonal elements of the kernel matrix
    diag_K = np.diag(K)
    
    # Compute the squared distances using the norm induced by K
    K_norm = diag_K[:, np.newaxis] + diag_K[np.newaxis, :] - 2 * K
    
    # Compute the Gaussian kernel
    K_gaussian = np.exp(-K_norm / (2 * sigma ** 2))
    
    return K_gaussian

def sw_matrix(X_left, X_right, sw=sm.SmithWaterman(), n_jobs=-1):
    """
    Compute the Smith-Waterman (SW) score matrix between two sets of sequences.

    Parameters
    ----------
    X_left : DataFrame
        A DataFrame containing sequences in the 'seq' column.
    X_right : DataFrame
        A DataFrame containing sequences in the 'seq' column.
    sw : object
        An object with a method `get_raw_score` to compute the SW score between two sequences.
    n_jobs : int, optional
        The number of jobs to run in parallel (default is -1, which means using all processors).

    Returns
    -------
    SW_matrix : ndarray
        A 2D array of shape (n_samples_left, n_samples_right) containing the SW scores.
    """
    SW_matrix = np.zeros((X_left.shape[0], X_right.shape[0]))

    def compute_score(i, j):
        return sw.get_raw_score(X_left['seq'].iloc[i], X_right['seq'].iloc[j])

    if X_left is X_right:
        indices = [(i, j) for i in range(X_left.shape[0]) for j in range(i, X_right.shape[0])]
        results = Parallel(n_jobs=n_jobs)(delayed(compute_score)(i, j) for i, j in tqdm(indices, desc=f"Computing SW matrix for {X_left.shape[0]} sequences"))
        for (i, j), score in zip(indices, results):
            SW_matrix[i, j] = score
            SW_matrix[j, i] = score
    else:
        indices = [(i, j) for i in range(X_left.shape[0]) for j in range(X_right.shape[0])]
        results = Parallel(n_jobs=n_jobs)(delayed(compute_score)(i, j) for i, j in tqdm(indices, desc=f"Computing SW matrix for {X_left.shape[0]} sequences"))
        for (i, j), score in zip(indices, results):
            SW_matrix[i, j] = score
    return normalize(SW_matrix)

def spectrum_kernel_matrix(X_left, X_right, k, n_jobs=-1):
    """
    Compute the spectrum kernel matrix between two sets of sequences.

    Parameters
    ----------
    X_left : DataFrame
        A DataFrame containing sequences in the 'seq' column.
    X_right : DataFrame
        A DataFrame containing sequences in the 'seq' column.
    k : int
        The length of k-mers to consider.
    n_jobs : int, optional
        The number of jobs to run in parallel (default is -1, which means using all processors).

    Returns
    -------
    K_spect : ndarray
        A 2D array of shape (n_samples_left, n_samples_right) containing the spectrum kernel values.
    """
    def get_kmers(s, k):
        """
        Returns a dictionary with the frequencies of k-mers in a sequence.

        Parameters
        ----------
        s : str
            The input sequence.
        k : int
            The length of k-mers.

        Returns
        -------
        kmers : Counter
            A Counter object with k-mer frequencies.
        """
        return Counter([s[i:i+k] for i in range(len(s) - k + 1)])
    
    def spectrum_kernel(s, t, k):
        """
        Computes the spectrum kernel between two sequences.

        Parameters
        ----------
        s : str
            The first sequence.
        t : str
            The second sequence.
        k : int
            The length of k-mers.

        Returns
        -------
        score : int
            The spectrum kernel score.
        """
        kmers_s = get_kmers(s, k)
        kmers_t = get_kmers(t, k)        
        common_kmers = set(kmers_s.keys()) & set(kmers_t.keys())
        return sum(kmers_s[kmer] * kmers_t[kmer] for kmer in common_kmers)
    
    def compute_score(i, j):
        return spectrum_kernel(X_left['seq'].iloc[i], X_right['seq'].iloc[j], k)
    
    K_spect = np.zeros((X_left.shape[0], X_right.shape[0]))
    
    if X_left is X_right:
        indices = [(i, j) for i in range(X_left.shape[0]) for j in range(i, X_right.shape[0])]
        results = Parallel(n_jobs=n_jobs)(delayed(compute_score)(i, j) for i, j in tqdm(indices, desc="Computing Spectrum Kernel Matrix"))
        for (i, j), score in zip(indices, results):
            K_spect[i, j] = score
            K_spect[j, i] = score
    else:
        indices = [(i, j) for i in range(X_left.shape[0]) for j in range(X_right.shape[0])]
        results = Parallel(n_jobs=n_jobs)(delayed(compute_score)(i, j) for i, j in tqdm(indices, desc="Computing Spectrum Kernel Matrix"))
        for (i, j), score in zip(indices, results):
            K_spect[i, j] = score    
    return normalize(K_spect)

def generate_mismatch_neighbors(kmer, alphabet, m):
    """
    Generate all k-mers within m mismatches of the given k-mer.
    Uses combinations (distinct positions) rather than product over positions.
    """
    neighbors = set()
    k = len(kmer)
    # For each combination of m distinct positions
    for positions in combinations(range(k), m):
        for replacements in product(alphabet, repeat=m):
            new_kmer = list(kmer)
            for pos, repl in zip(positions, replacements):
                new_kmer[pos] = repl
            neighbors.add("".join(new_kmer))
    # Always include the original k-mer
    neighbors.add(kmer)
    return neighbors

def compute_feature_vector(sequence, k, m, alphabet="ACGT", neighbor_cache=None):
    """
    Compute the mismatch kernel feature vector for a single sequence.
    Uses caching for neighbor generation.
    """
    if neighbor_cache is None:
        neighbor_cache = {}
    kmer_counts = defaultdict(np.uint16)  # Use np.uint16 to save memory
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        key = (kmer, m)
        if key not in neighbor_cache:
            neighbor_cache[key] = generate_mismatch_neighbors(kmer, alphabet, m)
        neighbors = neighbor_cache[key]
        for neighbor in neighbors:
            kmer_counts[neighbor] += 1
    return kmer_counts

def compute_mismatch_kernel(sequences, k=5, m=1, alphabet="ACGT"):
    """
    Compute the mismatch kernel matrix for a list of sequences.
    Parallelizes the feature vector computation.

    Parameters
    ----------
    sequences : list of str
        Input sequences.
    k : int, optional
        k-mer length (default 5).
    m : int, optional
        Maximum allowed mismatches (default 1).
    alphabet : str, optional
        The alphabet (default "ACGT").

    Returns
    -------
    K_normalized : ndarray
        The normalized kernel matrix.
    """
    n = len(sequences)
    # Create a shared neighbor cache (read-only after computed)
    neighbor_cache = {}
    
    # Compute feature vectors in parallel
    feature_vectors = list(Parallel(n_jobs=-1)(
        delayed(compute_feature_vector)(seq, k, m, alphabet, neighbor_cache)
        for seq in tqdm(sequences, total=n, desc="Computing feature vectors")
    ))
    
    # Build the global vocabulary from all feature vectors
    all_kmers = set()
    # Collect all k-mers from feature vectors in parallel
    all_kmers = set().union(*Parallel(n_jobs=-1)(
        delayed(lambda fv: set(fv.keys()))(fv) 
        for fv in tqdm(feature_vectors, desc="Collecting k-mers")
    ))
    all_kmers = sorted(all_kmers)
    kmer_index = {kmer: idx for idx, kmer in enumerate(all_kmers)}
    
    # Collect entries for the sparse matrix in parallel
    rows, cols, data = [], [], []
    results = Parallel(n_jobs=-1)(
        delayed(lambda i, fv: (
            [i] * len(fv), 
            [kmer_index[kmer] for kmer in fv.keys()], 
            list(fv.values())
        ))(i, fv) for i, fv in tqdm(enumerate(feature_vectors), total=len(feature_vectors), desc="Building sparse matrix entries")
    )
    
    for r, c, d in results:
        rows.extend(r)
        cols.extend(c)
        data.extend(d)
    
    # Build the sparse matrix (COO format) and convert to CSR
    X = coo_matrix((data, (rows, cols)), shape=(n, len(all_kmers)), dtype=np.float32).tocsr()
    
    # Compute kernel matrix (dot product)
    K = X.dot(X.T).toarray()
    
    return normalize(K)

def compute_mismatch_subkernel(sequences_left, sequences_right, k=5, m=1, alphabet="ACGT"):
    """
    Compute the mismatch kernel matrix for two lists of sequences.
    Parallelizes the feature vector computation.

    Parameters
    ----------
    sequences_left : list of str
        Input sequences for rows.
    sequences_right : list of str
        Input sequences for columns.
    k : int, optional
        k-mer length (default 5).
    m : int, optional
        Maximum allowed mismatches (default 1).
    alphabet : str, optional
        The alphabet (default "ACGT").

    Returns
    -------
    K_normalized : ndarray
        The normalized kernel matrix.
    """
    n_left = len(sequences_left)
    n_right = len(sequences_right)
    neighbor_cache = {}
    
    feature_vectors_left = list(Parallel(n_jobs=-1)(
        delayed(compute_feature_vector)(seq, k, m, alphabet, neighbor_cache)
        for seq in tqdm(sequences_left, total=n_left, desc="Computing feature vectors (left)")
    ))
    
    feature_vectors_right = list(Parallel(n_jobs=-1)(
        delayed(compute_feature_vector)(seq, k, m, alphabet, neighbor_cache)
        for seq in tqdm(sequences_right, total=n_right, desc="Computing feature vectors (right)")
    ))
    
    all_kmers = set().union(*Parallel(n_jobs=-1)(
        delayed(lambda fv: set(fv.keys()))(fv) 
        for fv in tqdm(feature_vectors_left + feature_vectors_right, desc="Collecting k-mers")
    ))
    all_kmers = sorted(all_kmers)
    kmer_index = {kmer: idx for idx, kmer in enumerate(all_kmers)}
    
    def build_sparse_matrix(feature_vectors, num_rows):
        rows, cols, data = [], [], []
        results = Parallel(n_jobs=-1)(
            delayed(lambda i, fv: (
                [i] * len(fv), 
                [kmer_index[kmer] for kmer in fv.keys()], 
                list(fv.values())
            ))(i, fv) for i, fv in tqdm(enumerate(feature_vectors), total=num_rows, desc="Building sparse matrix entries")
        )
        for r, c, d in results:
            rows.extend(r)
            cols.extend(c)
            data.extend(d)
        return coo_matrix((data, (rows, cols)), shape=(num_rows, len(all_kmers)), dtype=np.float32).tocsr()
    
    X_left = build_sparse_matrix(feature_vectors_left, n_left)
    X_right = build_sparse_matrix(feature_vectors_right, n_right)
    
    K = X_left.dot(X_right.T).toarray()
    
    return K

def LA_unit(x, y, beta, d, e):
    def S(a, b):
        return 1 if a == b else 0
    
    # Lengths of sequences
    len_x = len(x)
    len_y = len(y)
    
    # Initialize matrices with -inf for log(0) cases
    M = np.full((len_x + 1, len_y + 1), -np.inf)
    X = np.full((len_x + 1, len_y + 1), -np.inf)
    Y = np.full((len_x + 1, len_y + 1), -np.inf)
    X2 = np.full((len_x + 1, len_y + 1), -np.inf)
    Y2 = np.full((len_x + 1, len_y + 1), -np.inf)
    
    # Fill the matrices using logarithmic transformations
    for i in range(1, len_x + 1):
        for j in range(1, len_y + 1):
            # Compute log M(i,j) using log-sum-exp trick
            log_M_ij = beta * S(x[i-1], y[j-1])
            M[i, j] = log_M_ij + np.logaddexp(0, 
                                   np.logaddexp(M[i-1, j-1], 
                                   np.logaddexp(X[i-1, j-1], 
                                   Y[i-1, j-1])))
            
            # Compute log X(i,j)
            X[i, j] = np.logaddexp(beta*d + M[i-1, j], beta*e + X[i-1, j])
            
            # Compute log Y(i,j)
            Y[i, j] = np.logaddexp(beta*d + np.logaddexp(M[i, j-1], X[i, j-1]), beta*e + Y[i, j-1])
            
            # Compute log X2(i,j)
            X2[i, j] = np.logaddexp(M[i-1, j], X2[i-1, j])
            
            # Compute log Y2(i,j)
            Y2[i, j] = np.logaddexp(np.logaddexp(M[i, j-1], X2[i, j-1]), Y2[i, j-1])
    
    # Compute the log of the LA kernel
    log_K_LA = np.logaddexp(0,np.logaddexp(np.logaddexp(X2[len_x, len_y], Y2[len_x, len_y]), M[len_x, len_y]))
    
    return log_K_LA

def LA_kernel_matrix(X_left, X_right, beta=0.5, d=1, e=0.5, n_jobs=-1):
    """
    Compute the Logarithmic Alignment (LA) kernel matrix between two sets of sequences.

    Parameters
    ----------
    X_left : DataFrame
        A DataFrame containing sequences in the 'seq' column.
    X_right : DataFrame
        A DataFrame containing sequences in the 'seq' column.
    beta : float, optional
        The beta parameter for the LA kernel (default is 0.5).
    d : float, optional
        The d parameter for the LA kernel (default is 0.5).
    e : float, optional
        The e parameter for the LA kernel (default is 0.5).
    n_jobs : int, optional
        The number of jobs to run in parallel (default is -1, which means using all processors).

    Returns
    -------
    K_LA : ndarray
        A 2D array of shape (n_samples_left, n_samples_right) containing the LA kernel values.
    """
    def compute_score(i, j):
        return LA_unit(X_left['seq'].iloc[i], X_right['seq'].iloc[j], beta, d, e)
    
    K_LA = np.zeros((X_left.shape[0], X_right.shape[0]))
    
    if X_left is X_right:
        indices = [(i, j) for i in range(X_left.shape[0]) for j in range(i, X_right.shape[0])]
        results = Parallel(n_jobs=n_jobs)(delayed(compute_score)(i, j) for i, j in tqdm(indices, desc="Computing LA Kernel Matrix"))
        for (i, j), score in zip(indices, results):
            K_LA[i, j] = score
            K_LA[j, i] = score
    else:
        indices = [(i, j) for i in range(X_left.shape[0]) for j in range(X_right.shape[0])]
        results = Parallel(n_jobs=n_jobs)(delayed(compute_score)(i, j) for i, j in tqdm(indices, desc="Computing LA Kernel Matrix"))
        for (i, j), score in zip(indices, results):
            K_LA[i, j] = score
    return K_LA/beta


################################
### Kernel Summary Function  ###
################################

def compute_kernel_matrix(X_left, X_right, kernel, **kwargs):
    """
    Compute the kernel matrix between two sets of data using the specified kernel function.

    Parameters
    ----------
    X_left : ndarray or DataFrame
        The left set of data. If the kernel is 'mismatch' or 'LA', it should be a DataFrame with sequences in the 'seq' column.
    X_right : ndarray or DataFrame
        The right set of data. If the kernel is 'mismatch' or 'LA', it should be a DataFrame with sequences in the 'seq' column.
    kernel : str
        The type of kernel to use. Options are 'dist', 'exp', 'sw', 'spect', 'mismatch', 'LA', 'LA_gpu'.
    **kwargs : dict
        Additional parameters for the kernel function.

    Returns
    -------
    K : ndarray
        The computed kernel matrix.
    """
    if kernel == 'dist':
        return dist_kernel(X_left, X_right)
    elif kernel == 'exp':
        return exp_kernel(X_left, X_right, **kwargs)
    elif kernel == 'sw':
        return sw_matrix(X_left, X_right, **kwargs)
    elif kernel == 'spect':
        return spectrum_kernel_matrix(X_left, X_right, **kwargs)
    elif kernel == 'mismatch':
        return compute_mismatch_kernel(X_left['seq'], **kwargs)
    elif kernel == 'mis_sub':
        return compute_mismatch_subkernel(X_left['seq'], X_right['seq'], **kwargs)
    elif kernel == 'LA':
        return LA_kernel_matrix(X_left, X_right, **kwargs)
    
