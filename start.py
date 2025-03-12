import pandas as pd
import numpy as np
from Local_packages.run import KernelMethod, KernelMethodBias
from Local_packages.optimizer import SVM_solver, SVM_solver_with_bias


# Load the labels
Ytr0 = pd.read_csv('data/Ytr0.csv', index_col=0)
Ytr1 = pd.read_csv('data/Ytr1.csv', index_col=0)
Ytr2 = pd.read_csv('data/Ytr2.csv', index_col=0)
# Convert the labels to -1, 1
Ytr0 = 2*Ytr0['Bound'].values - 1
Ytr1 = 2*Ytr1['Bound'].values - 1
Ytr2 = 2*Ytr2['Bound'].values - 1

kernel_versions = ['5-1', '5-2', '6-1', '6-2', '7-1', '7-2', '8-1', '8-2', '9-1', '9-2', '10-2']
K_0_dict, K_1_dict, K_2_dict = {}, {}, {}

for version in kernel_versions:
    K_0_dict[version] = np.load(f'features/K_0_mismatch_{version}.npy')
    K_1_dict[version] = np.load(f'features/K_1_mismatch_{version}.npy')
    K_2_dict[version] = np.load(f'features/K_2_mismatch_{version}.npy')

### DATASET 1 ###
K_0 = K_0_dict['10-2'][:,:2000]**1.4
lambd_0 = 1e-2
method_0 = KernelMethodBias(K_0[:2000], Ytr0, lambd=lambd_0, solver=SVM_solver_with_bias)
method_0.train_test_split(test_size=0.001)
method_0.fit()

### DATASET 2 ###
K_1 = K_1_dict['10-2'][:,:2000]**1.5
lambd_1 = 1e-2
method_1 = KernelMethodBias(K_1[:2000], Ytr1, lambd=lambd_1, solver=SVM_solver_with_bias)
method_1.train_test_split(test_size=0.001)
method_1.fit()

### DATASET 3 ###
K_2 = K_2_dict['9-1'][:,:2000]**1.5
lambd_2 = 1e-2
method_2 = KernelMethodBias(K_2[:2000], Ytr2, lambd=lambd_2, solver=SVM_solver_with_bias)
method_2.train_test_split(test_size=0.001)
method_2.fit()

### Predictions ###
def predict_test_labels(K, method):
    K_te = K
    alpha = method.alpha
    b = method.b
    # Predictions
    Yte0 = np.sign(np.dot(K_te, alpha * method.Y[method.train_indices]) + b)
    return Yte0

Yte0 = predict_test_labels(K_0[2000:][:, method_0.train_indices], method_0)
Yte1 = predict_test_labels(K_1[2000:][:,method_1.train_indices], method_1)
Yte2 = predict_test_labels(K_2[2000:][:,method_2.train_indices], method_2)

Yte_file_name = 'Yte.csv'

# Concatenate and add Id column
Yte = np.concatenate([Yte0, Yte1, Yte2])
Yte = pd.DataFrame(data=(Yte + 1) // 2, columns=['Bound'], dtype='int64')
Yte.insert(0, 'Id', Yte.index)

# Save the predictions
Yte.to_csv(Yte_file_name, index=False)