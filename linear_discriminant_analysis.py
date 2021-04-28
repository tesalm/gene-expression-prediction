# -*- coding: utf-8 -*-

from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from parse_data import preprocess_data


if __name__ == '__main__':
    x_train, y_train, x_test = preprocess_data()

    components = [0, 1]
    solvers = ['svd', 'lsqr', 'eigen']
    param_grid = {'n_components': components, 'solver': solvers}

    LDA_clf = LinearDiscriminantAnalysis()

    estimator = GridSearchCV(LDA_clf, 
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
    # Best: 0.904199 using {'n_components': 0, 'solver': 'svd'}

