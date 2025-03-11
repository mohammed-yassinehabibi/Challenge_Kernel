import pandas as pd
import numpy as np
from Local_packages.run import KernelMethod
from Local_packages.optimizer import SVM_solver


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
K_0_concat = np.stack([K_0_dict[version][:, :2000] for version in ['8-2', '9-2', '5-1']], axis=0)
K_0_norm = np.exp(np.linalg.norm(K_0_concat, axis=0)-1)
K_0 = K_0_norm+1

lambd_0 = 1e-5
method_0 = KernelMethod(K_0[:2000, :2000], Ytr0, lambd=lambd_0, solver=SVM_solver)
method_0.train_test_split(test_size=0.1, random_state=42)
method_0.fit()

### DATASET 2 ###
K_1 = K_1_dict['9-2'][:,:2000]**2

lambd_1 = 1e-4
method_1 = KernelMethod((K_1+1)[:2000], Ytr1, lambd=lambd_1, solver=SVM_solver)
method_1.train_test_split(test_size=0.1, random_state=10)
method_1.fit()

### DATASET 3 ###
K_2 = K_2_dict['9-1'][:,:2000]*K_2_dict['9-2'][:,:2000]**2+1

lambd_2 = 2e-4
method_2 = KernelMethod((K_2)[:2000, :2000], Ytr2, lambd=lambd_2, solver=SVM_solver)
method_2.train_test_split(test_size=0.1, random_state=10)
method_2.fit()

### Predictions ###
def predict_test_labels(K, method):
    K_te = K
    alpha = method.alpha
    Yte0 = np.sign(K_te @ alpha)
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