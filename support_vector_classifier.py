# -*- coding: utf-8 -*-

import numpy as np

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC

from parse_data import preprocess_data


if __name__ == '__main__':
    x_train, y_train, x_test = preprocess_data()

    components = np.arange(20, 120, 20)   # [20, 40, 60, ..., 100]
    C_range = 10.0 ** np.arange(-5, 1, 1) # [1e-04, 1e-03, ..., 1]
    kernels = ['linear', 'rbf', 'poly', 'sigmoid'] # for kernel SVC

    # The implementation is based on libsvm. The fit time scales at least quadratically 
    # with the number of samples and may be impractical beyond tens of thousands of samples. 
    # For large datasets consider using LinearSVC
    kernelSVC_clf = SVC(probability=False) # default SVC(C=1.0, kernel='rbf')

    # Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear 
    # rather than libsvm, so it has more flexibility in the choice of penalties and loss functions 
    # and should scale better to large numbers of samples
    LinearSVC_clf = LinearSVC(dual=False)

    pca = PCA()
    pipe = Pipeline(steps=[('pca', pca), ('SVC_clf', LinearSVC_clf)])

    estimator = GridSearchCV(pipe,
                             dict(pca__n_components = components,
                             #SVC_clf__kernel = kernels,
                             SVC_clf__C = C_range),
                             scoring='roc_auc', cv = 5, verbose=3)

    estimator.fit(x_train, y_train)

    predictions = estimator.predict(x_test)


    # GridSearch parameters
    best_params = estimator.best_params_
    best_score = estimator.best_score_
    best_index = estimator.best_index_
    cv_results = estimator.cv_results_


    # kernelSVC_clf.probability = True
    # kernelSVC_clf.C = best_params['SVC_clf__C']
    # kernelSVC_clf.kernel = best_params['SVC_clf__kernel']
    # pca.components = best_params['pca__n_components']
    # pipe = Pipeline(steps=[('pca', pca), ('SVC_clf', kernelSVC_clf)])
    # pipe.fit(x_train, y_train)
    # pred_prob = pipe.predict_proba(x_test)[:,1]

    print("Best: %f using %s" % (best_score, best_params))
    # Best: 0.910880 using {'SVC_clf__C': 0.01, 'pca__n_components': 60}

