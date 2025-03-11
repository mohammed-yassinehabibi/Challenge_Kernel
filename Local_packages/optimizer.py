import numpy as np
import osqp
from tqdm import tqdm
from scipy.sparse import csc_matrix
from scipy import optimize

##################################
### Kernel Logistic Regression ###
##################################

def KLR_solver(K, Y, lambd, tol=1e-6, max_iter=10, Y_test=None, K_test=None):
    """
    Solve Kernel Logistic Regression (KLR) by Iteratively Reweighted Least Squares (IRLS) on the quadratic loss.

    Parameters
    ----------
    K : ndarray
        Kernel matrix of shape (n_samples, n_samples).
    Y : ndarray
        Labels of shape (n_samples,).
    lambd : float
        Regularization parameter.
    tol : float, optional
        Tolerance for stopping criterion. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 10.
    Y_test : ndarray, optional
        Test labels of shape (n_test_samples,). Default is None.
    K_test : ndarray, optional
        Test kernel matrix of shape (n_test_samples, n_samples). Default is None.

    Returns
    -------
    alpha_t : ndarray
        Solution vector of shape (n_samples,).
    """
    def log_loss(Y, m_t, alpha, lambd):
        """
        Compute the logistic loss.

        Parameters
        ----------
        Y : ndarray
            Labels of shape (n_samples,).
        m_t : ndarray
            Predicted values of shape (n_samples,).
        alpha : ndarray
            Solution vector of shape (n_samples,).
        lambd : float
            Regularization parameter.

        Returns
        -------
        float
            Logistic loss value.
        """
        n = Y.shape[0]
        loss = np.sum(np.log(1 + np.exp(-Y * m_t)))
        return lambd * np.dot(alpha, m_t) / 2 + loss / n

    def log_loss_2(Y, K, alpha, lambd):
        """
        Compute the logistic loss using kernel matrix.

        Parameters
        ----------
        Y : ndarray
            Labels of shape (n_samples,).
        K : ndarray
            Kernel matrix of shape (n_samples, n_samples).
        alpha : ndarray
            Solution vector of shape (n_samples,).
        lambd : float
            Regularization parameter.

        Returns
        -------
        float
            Logistic loss value.
        """
        n = Y.shape[0]
        m_t = np.dot(K, alpha)
        loss = np.sum(np.log(1 + np.exp(-Y * m_t)))
        return lambd * np.dot(alpha, m_t) / 2 + loss / n

    def sigmoid(u):
        """
        Compute the sigmoid function.

        Parameters
        ----------
        u : ndarray
            Input array.

        Returns
        -------
        ndarray
            Sigmoid values.
        """
        return 1 / (1 + np.exp(-u))

    def W_func(Y, m_t):
        """
        Compute the weight matrix for IRLS.

        Parameters
        ----------
        Y : ndarray
            Labels of shape (n_samples,).
        m_t : ndarray
            Predicted values of shape (n_samples,).

        Returns
        -------
        ndarray
            Diagonal weight matrix.
        """
        W = np.zeros(Y.shape[0])
        W = sigmoid(m_t) * sigmoid(-m_t)
        return np.diag(W)

    def z_func(Y, m_t):
        """
        Compute the z vector for IRLS.

        Parameters
        ----------
        Y : ndarray
            Labels of shape (n_samples,).
        m_t : ndarray
            Predicted values of shape (n_samples,).

        Returns
        -------
        ndarray
            z vector.
        """
        return m_t + Y / sigmoid(Y * m_t)

    n = K.shape[0]
    alpha_t = np.zeros(K.shape[0])
    criterion = np.inf
    for i in tqdm(range(max_iter), desc="IRLS Progress"):
        m_t = np.dot(K, alpha_t)
        accuracy = np.mean(np.sign(m_t) == Y)
        if Y_test is not None and K_test is not None:
            test_accuracy = np.mean(np.sign(np.dot(K_test, alpha_t)) == Y_test)
        W_t = W_func(Y, m_t)
        z_t = z_func(Y, m_t)
        W_t_sqrt = np.sqrt(W_t)
        A = (W_t_sqrt @ K @ W_t_sqrt + lambd * n * np.eye(n)) @ np.diag(1 / np.diag(W_t_sqrt))
        B = W_t_sqrt @ z_t
        alpha_new = np.linalg.solve(A, B)
        criterion = np.linalg.norm(alpha_new - alpha_t)
        if criterion < tol:
            accuracy = np.mean(np.sign(np.dot(K, alpha_new)) == Y)
            if Y_test is not None and K_test is not None:
                test_accuracy = np.mean(np.sign(np.dot(K_test, alpha_new)) == Y_test)
            break
        alpha_t = alpha_new
    return alpha_t

##################
### Kernel SVM ###
##################

def svm_dual_objective(K, Y, alpha):
    """
    Compute the dual objective for SVM.

    Parameters
    ----------
    K : ndarray
        Kernel matrix of shape (n_samples, n_samples).
    Y : ndarray
        Labels of shape (n_samples,).
    alpha : ndarray
        Solution vector of shape (n_samples,).

    Returns
    -------
    float
        Dual objective value.
    """
    return 2 * alpha @ Y - np.dot(alpha, np.dot(K, alpha))

def SVM_solver(K, Y, lambd, is_2_svm=False):
    """
    Solve the quadratic problem for SVM.

    Parameters
    ----------
    K : ndarray
        Kernel matrix of shape (n_samples, n_samples).
    Y : ndarray
        Labels of shape (n_samples,).
    lambd : float
        Regularization parameter.
    is_2_svm : bool, optional
        Flag to indicate if it is 2-class SVM. Default is False.

    Returns
    -------
    ndarray
        Solution vector of shape (n_samples,).
    """
    solver = osqp.OSQP()
    n = K.shape[0]
    if is_2_svm:
        K_sparse = csc_matrix(K + lambd * n * np.eye(n))
        solver.setup(P=K_sparse, q=-Y, A=csc_matrix(np.diag(Y)), l=np.zeros(n), u=np.inf * np.ones(n), verbose=False)
    else:
        K_sparse = csc_matrix(K)
        solver.setup(P=K_sparse, q=-Y, A=csc_matrix(np.diag(Y)), l=np.zeros(n), u=np.ones(n) / (2 * n * lambd), verbose=False)
    results = solver.solve()
    return results.x

import osqp
import numpy as np
from scipy.sparse import csc_matrix

import osqp
import numpy as np
from scipy.sparse import csc_matrix, vstack, eye

def SVM_solver_with_bias(K, Y, lambd):
    """
    Solves the standard SVM dual problem with a bias term computed via KKT conditions.
    
    Parameters
    ----------
    K : ndarray
        Kernel matrix of shape (n_samples, n_samples).
    Y : ndarray
        Labels of shape (n_samples,). Expected to be -1 or +1.
    C : float
        Regularization parameter (box constraint).
    
    Returns
    -------
    alpha : ndarray
        Dual variables of shape (n_samples,).
    b : float
        Bias term.
    """
    C = 1 / (2 * lambd)
    n = K.shape[0]
    
    # Build the QP problem corresponding to:
    #   min (1/2) * alpha^T (Y*Y^T ∘ K) alpha - 1^T alpha
    # subject to 0 <= alpha_i <= C and sum_i alpha_i Y_i = 0.
    Q = csc_matrix(np.outer(Y, Y) * K)
    q = -np.ones(n)
    
    # To handle both the box constraints and the equality constraint,
    # we create a constraint matrix A such that:
    #   A = [ I ]
    #       [ Y^T ]
    # and set bounds for the first n rows (box constraints) and one equality row.
    A_box = eye(n, format='csc')
    A_eq = csc_matrix(Y.reshape(1, -1))
    A = vstack([A_box, A_eq]).tocsc()
    
    # Lower and upper bounds:
    # For the box constraints: 0 <= alpha_i <= C.
    # For the equality constraint: sum_i alpha_i Y_i = 0.
    l_box = np.zeros(n)
    u_box = C * np.ones(n)
    
    l_eq = np.array([0.0])
    u_eq = np.array([0.0])
    
    l = np.hstack([l_box, l_eq])
    u = np.hstack([u_box, u_eq])
    
    # Setup and solve the quadratic program with OSQP
    solver = osqp.OSQP()
    solver.setup(P=Q, q=q, A=A, l=l, u=u, verbose=False)
    results = solver.solve()
    alpha = results.x
    
    # Compute bias b using the KKT conditions on the support vectors.
    # We select support vectors with 0 < alpha < C (up to a tolerance)
    tol = 1e-5
    sv_idx = np.where((alpha > tol) & (alpha < C - tol))[0]
    if len(sv_idx) > 0:
        b = np.mean(Y[sv_idx] - (np.outer(alpha * Y, np.ones(n))[sv_idx] * K[sv_idx]).sum(axis=1))
    else:
        # Fallback: if no strict support vector, use all vectors with alpha > tol.
        sv_idx = np.where(alpha > tol)[0]
        b = np.mean(Y[sv_idx] - (np.outer(alpha * Y, np.ones(n))[sv_idx] * K[sv_idx]).sum(axis=1))
    
    return alpha, b
