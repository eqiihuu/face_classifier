#! /usr/bin/python

'''   SVM  Facial Expression recognition.   '''

import time
import os
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA


# Load data
def load_data(path, file):
    src_path = os.path.join(path, file)
    data = pd.read_csv(src_path, skiprows=[0])
    images = np.array(data.iloc[:, 1:-1])
    labels = np.array(data.iloc[:, 0:1])
    num, _ = labels.shape
    labels = labels.reshape(num)
    return images, labels


# PCA
def pca(data_mat, n):
    mean_val = np.mean(data_mat, axis=0)  # Get the means of each feature (column)
    new_data = data_mat-mean_val   # Get zero-mean features
    cov_mat = np.cov(new_data, rowvar=0)  # Get covariance
    eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))  # Get eigen_values and eigen_vectors
    eig_val_indice = np.argsort(eig_vals)  # Sort the eigen_values (asc)
    n_eig_val_indice = eig_val_indice[-1:-(n+1):-1]  # Get the largest n eigen_values'index
    n_eig_vect = eig_vects[:, n_eig_val_indice]  # Get the largest n eigen_values' eigen_vectors
    lowd_data_mat = new_data*n_eig_vect
    # recon_mat = (lowd_data_mat*n_eig_vect.T)+mean_val
    return np.array(lowd_data_mat), n_eig_vect


# Get PCA with given eigen vector
def pca_vect(data_mat, n_eig_vect):
    mean_val = np.mean(data_mat, axis=0)  # Get the means of each feature (column)
    new_data = data_mat-mean_val   # Get zero-mean features
    lowd_data_mat = new_data*n_eig_vect
    # recon_mat = (lowd_data_mat*n_eig_vect.T)+mean_val
    return np.array(lowd_data_mat)

# Load settings
load = 0  # load pretrained model

# PCA parameters
n = 100  # PCA feature number

# SVM parameters
C = 1  # 2.67
gamma = 1.0/n  # 5.383
# Larger C: more general model (Penalty parameter C of the error term)
# Larger gamma: less support vectors

# path of data directory
path = '/Users/qihucn/Documents/EE576/Project/faceDetect'
print('loading data...')
train_data, train_label = load_data(path, 'train.csv')
print '  Number of train images: ', train_label.shape
test_data, test_label = load_data(path, 'test.csv')
print '  Number of test images: ', test_label.shape

print('Computing PCA ...')
# # Manually PCA
# train_sample, eigen_vect = pca(train_data, n)
# test_sample = pca_vect(test_data, eigen_vect)

# PCA in sklearn
pca = PCA(n_components=n)
pca.fit(train_data)
train_sample = pca.transform(train_data)
test_sample = pca.transform(test_data)

version = 'PCA'+str(n)+'_'+str(C)+'_'+str(gamma)
f = open('sk_svm_result_'+version+'.txt', 'w')

if load == 1:
    clf = joblib.load('sklearn_svm_model'+'_PCA'+str(n)+'_'+str(C)+'_'+str(gamma)+'.pkl')
else:

    t1 = time.time()
    print 'training SVM...'
    clf = svm.SVC(C=C, gamma=gamma, verbose=False, class_weight='balanced')
    clf.fit(train_sample, train_label)
    joblib.dump(clf, 'sklearn_svm_model'+'_PCA'+str(n)+'_'+str(C)+'_'+str(gamma)+'.pkl')
    t2 = time.time()
    s1 = 'Time: Train: %.3f s' % (t2-t1)
    f.write(s1+'\n')
    print s1
t3 = time.time()
print 'testing SVM...'
accuracy = clf.score(test_sample, test_label)
predicts = clf.predict(test_sample)
conf_mat = confusion_matrix(test_label, predicts)
print predicts
t4 = time.time()
s2 = 'Accuracy: %.2f %%' % (accuracy*100)
s3 = 'Time: Test: %.3f s' % (t4-t3)
f.write(s3+'\n'+s2+'\n')
print s3
print s2
print conf_mat
f.close()
