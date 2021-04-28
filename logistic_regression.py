# -*- coding: utf-8 -*-

import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from parse_data import preprocess_data


if __name__ == '__main__':
    x_train, y_train, x_test = preprocess_data()

    # Train a classiﬁer with the training data

    C_range = 10.0 ** np.arange(-5, 1, 1) # [1e-05, 1e-04, ..., 1]
    best_C = 0; best_score = 0; best_penalty = ""
    scores = []

    LR_clf = LogisticRegression(solver="liblinear")

    # 10x cross validation with different C and penalty term values 

    print('C          | L1 Penalty | L2 Penalty')
    print('-----------|------------|-----------')

    for C in C_range:
        LR_clf.C = C
        LR_clf.penalty = "l1"
        L1_score = cross_val_score(LR_clf, x_train, y_train, 
                                   cv=10, scoring='roc_auc').mean()           
        LR_clf.penalty = "l2"
        L2_score = cross_val_score(LR_clf, x_train, y_train, 
                                   cv=10, scoring='roc_auc').mean()

        if L1_score > best_score or L2_score > best_score:
            best_C = C
            if L1_score > L2_score:
                best_score = L1_score
                best_penalty = "l1"
            else:
                best_score = L2_score
                best_penalty = "l2"

        scores.append( [C, L1_score, L2_score] )
        print('{:.3E}  | {:.5f}    | {:.5f}'
              .format(C, L1_score, L2_score)) 

    accuracies = np.asarray(scores) # list to array conversion

    LR_clf.penalty = best_penalty
    LR_clf.C = best_C
    LR_clf.fit(x_train, y_train)
    pred = LR_clf.predict(x_test)

    # remove the first column (probability of the gene not being active)
    pred_prob = LR_clf.predict_proba(x_test)[:,1] # probabilitys of the gene being active


    # create submission csv file    
    with open('subm.csv','wb') as file:
        GeneId = 1
        for row in pred_prob:
            if GeneId == 1:
                file.write(b'GeneId,Prediction\n')      
            file.write(str(GeneId).encode() + b',' + str(round(row,4)).encode() + b'\n')
            GeneId = GeneId + 1
    file.close()

