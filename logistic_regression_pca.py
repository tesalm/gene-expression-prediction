# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from parse_data import preprocess_data


if __name__ == '__main__':
    x_train, y_train, x_test = preprocess_data()

    C_range = 10.0 ** np.arange(-5, 1, 1) # [1e-05, 1e-04, ..., 1]
    components = np.arange(10, 110, 10)   # [10, 20, 30, ..., 100]

    LR_clf = LogisticRegression(solver="liblinear")
    pca = PCA()
    pipe = Pipeline(steps=[('pca', pca), ('LR_clf', LR_clf)])

    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    estimator = GridSearchCV(pipe,
                             dict(pca__n_components = components,
                             LR_clf__C = C_range, 
                             LR_clf__penalty = ["l1", "l2"]),
                             scoring = 'roc_auc', cv = 10, verbose=3)

    estimator.fit(x_train, y_train)

    # Predict probabilities on the estimator with the best found parameters
    # and remove the first column (probability of the gene NOT being active)
    pred_prob = estimator.predict_proba(x_test)[:,1]


    # Plot the PCA spectrum
    pca.fit(x_train)
    plt.figure(1, figsize=(8, 5))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')
    plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
                linestyle=':', label='n_components chosen')
    plt.legend(prop=dict(size=12))

    # GridSearch parameters
    best_params = estimator.best_params_
    best_score = estimator.best_score_
    best_index = estimator.best_index_
    cv_results = estimator.cv_results_

    print("Best: %f using %s" % (best_score, best_params))
    # Best: 0.911474 using {'LR_clf__C': 0.01, 'LR_clf__penalty': 'l1', 'pca__n_components': 50}

