from Local_packages.optimizer import KLR_solver, SVM_solver
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

class KernelMethod():
    """
    A class used to represent a Kernel Method for classification tasks.
    Attributes
    ----------
    K : np.ndarray
        The kernel matrix.
    Y : np.ndarray
        The target labels.
    lambd : float, optional
        The regularization parameter (default is 1).
    solver : function, optional
        The solver function to use for fitting the model (default is KLR_solver).
    kwargs : dict
        Additional keyword arguments to pass to the solver function.
    test_indices : np.ndarray or None
        Indices of the test set after splitting the data.
    train_indices : np.ndarray or None
        Indices of the train set after splitting the data.
    alpha : np.ndarray or None
        The solution vector after fitting the model.
    Methods
    -------
    split_data(test_size=0.2)
        Splits the data into training and testing sets.
    fit()
        Fits the model using the training data.
    evaluate()
        Evaluates the model on both training and testing data and returns the accuracies.
    """
    def __init__(self, K, Y, lambd=1, solver=KLR_solver, **kwargs):
        self.kernel = K
        self.Y = Y
        self.lambd = lambd
        self.solver = solver
        self.kwargs = kwargs
        self.test_indices = None
        self.train_indices = None
        self.alpha = None
    
    def train_test_split(self, test_size=0.2, random_state=None):
        """
        Splits the data into training and testing sets.
        
        Parameters
        ----------
        test_size : float, optional
            The proportion of the dataset to include in the test split (default is 0.2).
        random_state : int or None, optional
            The seed used by the random number generator (default is None).
        """
        n = self.Y.shape[0]
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.random.permutation(n)
        split = int(n * test_size)
        self.test_indices, self.train_indices = indices[:split], indices[split:]

    def fit(self):
        """
        Fits the model using the training data.
        """
        K_train = self.kernel[self.train_indices][:, self.train_indices]
        Y_train = self.Y[self.train_indices]
        self.alpha = self.solver(K_train, Y_train, self.lambd, **self.kwargs)

    def evaluate(self):
        """
        Evaluates the model on both training and testing data and returns the accuracies.
        
        Returns
        -------
        train_accuracy : float
            The accuracy of the model on the training data.
        test_accuracy : float
            The accuracy of the model on the testing data.
        """
        K_train = self.kernel[self.train_indices][:, self.train_indices]
        Y_train = self.Y[self.train_indices]
        m_train = np.dot(K_train, self.alpha)
        train_accuracy = np.mean(np.sign(m_train) == Y_train)
        
        K_test = self.kernel[self.test_indices][:, self.train_indices]
        Y_test = self.Y[self.test_indices]
        m_test = np.dot(K_test, self.alpha)
        test_accuracy = np.mean(np.sign(m_test) == Y_test)
        
        return train_accuracy, test_accuracy
    
    def cross_validate(self, n_folds=5, test_size=0.2):
        """
        Performs cross-validation on the model.
        
        Parameters
        ----------
        n_folds : int, optional
            The number of folds to use for cross-validation (default is 5).
        test_size : float, optional
            The proportion of the dataset to include in the test split (default is 0.2).
        
        Returns
        -------
        accuracies : list
            A list of accuracies for each fold.
        """
        n = self.Y.shape[0]
        indices = np.random.permutation(n)
        fold_size = int(n * test_size)
        accuracies = []
        for i in range(n_folds):
            test_indices = indices[i * fold_size:(i + 1) * fold_size]
            train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
            K_train = self.kernel[train_indices][:, train_indices]
            Y_train = self.Y[train_indices]
            alpha = self.solver(K_train, Y_train, self.lambd, **self.kwargs)
            m_test = np.dot(self.kernel[test_indices][:, train_indices], alpha)
            accuracy = np.mean(np.sign(m_test) == self.Y[test_indices])
            accuracies.append(accuracy)
        return accuracies
    
    def grid_search(self, lambdas, n_folds=5, test_size=0.2):
        """
        Performs grid search on the regularization parameter.
        
        Parameters
        ----------
        lambdas : list
            A list of regularization parameters to search over.
        n_folds : int, optional
            The number of folds to use for cross-validation (default is 5).
        
        Returns
        -------
        best_lambda : float
            The best regularization parameter found.
        best_accuracy : float
            The best accuracy found.
        """
        best_lambda = None
        best_accuracy = 0
        for lambd in lambdas:
            self.lambd = lambd
            accuracies = self.cross_validate(n_folds, test_size)
            accuracy = np.mean(accuracies)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_lambda = lambd
            print(f"Lambda: {lambd}, Accuracy: {accuracy:.2f}")
        return best_lambda, round(best_accuracy, 4)*100
    
    def validate(self, test_size=0.1, n_splits=10, n_jobs=-1):
        def single_split_evaluation(_):
            self.train_test_split(test_size=test_size)
            self.fit()
            return self.evaluate()[1]

        accuracies = Parallel(n_jobs=n_jobs)(delayed(single_split_evaluation)(i) for i in tqdm(range(n_splits)))
        average_accuracy = np.mean(accuracies)
        print(f'Average Accuracy: {average_accuracy}')
        min_accuracy = np.min(accuracies)
        max_accuracy = np.max(accuracies)
        print(f'Min Accuracy: {min_accuracy}', f'Max Accuracy: {max_accuracy}')
        return average_accuracy
    
    
class KernelMethodBias():
    """
    A class used to represent a Kernel Method for classification tasks.
    Attributes
    ----------
    K : np.ndarray
        The kernel matrix.
    Y : np.ndarray
        The target labels.
    lambd : float, optional
        The regularization parameter (default is 1).
    solver : function, optional
        The solver function to use for fitting the model (default is KLR_solver).
    kwargs : dict
        Additional keyword arguments to pass to the solver function.
    test_indices : np.ndarray or None
        Indices of the test set after splitting the data.
    train_indices : np.ndarray or None
        Indices of the train set after splitting the data.
    alpha : np.ndarray or None
        The solution vector after fitting the model.
    Methods
    -------
    split_data(test_size=0.2)
        Splits the data into training and testing sets.
    fit()
        Fits the model using the training data.
    evaluate()
        Evaluates the model on both training and testing data and returns the accuracies.
    """
    def __init__(self, K, Y, lambd=1, solver=KLR_solver, **kwargs):
        self.kernel = K
        self.Y = Y
        self.lambd = lambd
        self.solver = solver
        self.kwargs = kwargs
        self.test_indices = None
        self.train_indices = None
        self.alpha = None
        self.b = 0
    
    def train_test_split(self, test_size=0.2, random_state=None):
        """
        Splits the data into training and testing sets.
        
        Parameters
        ----------
        test_size : float, optional
            The proportion of the dataset to include in the test split (default is 0.2).
        random_state : int or None, optional
            The seed used by the random number generator (default is None).
        """
        n = self.Y.shape[0]
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.random.permutation(n)
        split = int(n * test_size)
        self.test_indices, self.train_indices = indices[:split], indices[split:]

    def fit(self):
        """
        Fits the model using the training data.
        """
        K_train = self.kernel[self.train_indices][:, self.train_indices]
        Y_train = self.Y[self.train_indices]
        self.alpha, self.b = self.solver(K_train, Y_train, self.lambd, **self.kwargs)

    def evaluate(self):
        """
        Evaluates the model on both training and testing data and returns the accuracies.
        
        Returns
        -------
        train_accuracy : float
            The accuracy on the training data.
        test_accuracy : float
            The accuracy on the testing data.
        """
        # Training evaluation
        K_train = self.kernel[self.train_indices][:, self.train_indices]
        Y_train = self.Y[self.train_indices]
        m_train = np.dot(K_train, self.alpha * self.Y[self.train_indices]) + self.b
        train_accuracy = np.mean(np.sign(m_train) == Y_train)

        # Testing evaluation
        K_test = self.kernel[self.test_indices][:, self.train_indices]
        Y_test = self.Y[self.test_indices]
        m_test = np.dot(K_test, self.alpha * self.Y[self.train_indices]) + self.b
        test_accuracy = np.mean(np.sign(m_test) == Y_test)

        return train_accuracy, test_accuracy
    
    def cross_validate(self, n_folds=5, test_size=0.2):
        """
        Performs cross-validation on the model.
        
        Parameters
        ----------
        n_folds : int, optional
            The number of folds to use for cross-validation (default is 5).
        test_size : float, optional
            The proportion of the dataset to include in the test split (default is 0.2).
        
        Returns
        -------
        accuracies : list
            A list of accuracies for each fold.
        """
        n = self.Y.shape[0]
        indices = np.random.permutation(n)
        fold_size = int(n * test_size)
        accuracies = []
        for i in range(n_folds):
            test_indices = indices[i * fold_size:(i + 1) * fold_size]
            train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
            K_train = self.kernel[train_indices][:, train_indices]
            Y_train = self.Y[train_indices]
            alpha, b = self.solver(K_train, Y_train, self.lambd, **self.kwargs)
            m_test = np.dot(self.kernel[test_indices][:, train_indices], alpha * Y_train) + b
            accuracy = np.mean(np.sign(m_test) == self.Y[test_indices])
            accuracies.append(accuracy)
        return accuracies
    
    def grid_search(self, lambdas, n_folds=5, test_size=0.2):
        """
        Performs grid search on the regularization parameter.
        
        Parameters
        ----------
        lambdas : list
            A list of regularization parameters to search over.
        n_folds : int, optional
            The number of folds to use for cross-validation (default is 5).
        
        Returns
        -------
        best_lambda : float
            The best regularization parameter found.
        best_accuracy : float
            The best accuracy found.
        """
        best_lambda = None
        best_accuracy = 0
        for lambd in lambdas:
            self.lambd = lambd
            accuracies = self.cross_validate(n_folds, test_size)
            accuracy = np.mean(accuracies)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_lambda = lambd
            print(f"Lambda: {lambd}, Accuracy: {accuracy:.2f}")
        return best_lambda, round(best_accuracy, 4)*100
    
    def validate(self, test_size=0.1, n_splits=10, n_jobs=-1):
        def single_split_evaluation(_):
            self.train_test_split(test_size=test_size)
            self.fit()
            return self.evaluate()[1]

        accuracies = Parallel(n_jobs=n_jobs)(delayed(single_split_evaluation)(i) for i in tqdm(range(n_splits)))
        average_accuracy = np.mean(accuracies)
        print(f'Average Accuracy: {average_accuracy}')
        min_accuracy = np.min(accuracies)
        max_accuracy = np.max(accuracies)
        print(f'Min Accuracy: {min_accuracy}', f'Max Accuracy: {max_accuracy}')
        return average_accuracy
    
    
