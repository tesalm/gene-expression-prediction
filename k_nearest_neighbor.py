# -*- coding: utf-8 -*-

import numpy as np

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from parse_data import preprocess_data


if __name__ == '__main__':
    x_train, y_train, x_test = preprocess_data()

    KNN_clf = KNeighborsClassifier() # default neighbors = 5
    pca = PCA()
    pipe = Pipeline(steps=[('pca', pca), ('KNN_clf', KNN_clf)])

    components = np.arange(20, 120, 20)   # [20, 40, 60, ..., 100]
    neighbors = [3, 4, 5, 6, 7, 8, 9, 10]
    weights = ['distance', 'uniform']
    algorithm =  ['ball_tree', 'kd_tree', 'brute']
    metric = ['euclidean', 'minkowski']

    estimator = GridSearchCV(pipe,
                             dict(pca__n_components = components,
                             KNN_clf__weights = weights,
                             KNN_clf__algorithm = algorithm,
                             KNN_clf__metric = metric,
                             KNN_clf__n_neighbors = neighbors),
                             scoring = 'roc_auc', cv = 5, verbose=3)

    estimator.fit(x_train, y_train)

    # Predict probabilities on the estimator with the best found parameters
    # and remove the first column (probability of the gene NOT being active)
    pred_prob = estimator.predict_proba(x_test)[:,1]

    # GridSearch parameters
    best_params = estimator.best_params_
    best_score = estimator.best_score_
    best_index = estimator.best_index_
    cv_results = estimator.cv_results_

    print("Best: %f using %s" % (best_score, best_params))
    # Best: 0.897877 using {'KNN_clf__algorithm': 'kd_tree', 'KNN_clf__metric': 'minkowski', 'KNN_clf__n_neighbors': 10, 'KNN_clf__weights': 'uniform'}

