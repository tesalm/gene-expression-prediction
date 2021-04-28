# -*- coding: utf-8 -*-

import numpy as np

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import (AdaBoostClassifier, 
                              ExtraTreesClassifier, 
                              RandomForestClassifier,
                              GradientBoostingClassifier)

from parse_data import preprocess_data


if __name__ == '__main__':
    x_train, y_train, x_test = preprocess_data()

    components = np.arange(10, 110, 10)  # [10, 20, 30, ..., 100]
    estimators = np.arange(10, 160, 10)  # [10, 20, 30, ..., 150]
    param_grid = {'pca__n_components': components, 'clf__n_estimators': estimators}

    pca = PCA()
    RF_clf = RandomForestClassifier()
    ERT_clf = ExtraTreesClassifier()
    AB_clf = AdaBoostClassifier()
    GBT_clf = GradientBoostingClassifier()

    # classifiers = [("Random Forest", RF_clf), 
    #                ("Extra-Trees", ERT_clf), 
    #                ("AdaBoost", AB_clf), 
    #                ("GB-Trees", GBT_clf)]

    pipe = Pipeline(steps=[('pca', pca), ('clf', AB_clf)])

    estimator = GridSearchCV(pipe, 
                             param_grid,
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
    # Best: 0.908123 using {'AB_clf__n_estimators': 110, 'pca__n_components': 20}

