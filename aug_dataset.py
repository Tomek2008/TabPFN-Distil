import numpy as np

def aug_dataset(num_times, X_train, rand_mask=0.8, float_var=1, new_class_bal=0.5):
    X_aug = X_train.copy()
    for _ in range(num_times):
        aug_size = round(len(X_train)*1)
        rand_mask = np.random.rand(aug_size, len(X_train[0])) > 0.8

        one_hot_mask = np.logical_or(X_train==1.0, X_train==0.0)

        rep_val = np.random.randn(aug_size, len(X_train[0])) * float_var

        X_aug_new = one_hot_mask * ( X_train*(1-rand_mask) + (rep_val>0.5)*rand_mask )
        X_aug_new += (1-one_hot_mask) * (X_train + rep_val*rand_mask)

        X_aug = np.concat([X_aug, X_aug_new])
    return X_aug