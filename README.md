# Gene Expression Prediction
### Predicting gene expression from histone modification signals.

## Overview
Histone modifications are playing an important role in affecting gene regulation. Nowadays, predicting gene expression from histone modification signals is a widely studied research topic.

The dataset is on "E047" (Primary T CD8+ naive cells from peripheral blood) celltype from Roadmap Epigenomics Mapping Consortium (REMC) database. For each gene, it has 100 bins with five core histone modification marks [1]. (We divide the 10,000 basepair(bp) DNA region (+/-5000bp) around the transcription start site (TSS) of each gene into bins of length 100 bp [2], and then count the reads of 100 bp in each bin. Finally, the signal of each gene has a shape of 100x5.)

The goal of this competition is to develop algorithms for accurate predicting gene expression level. High gene expression level corresponds to target label = 1, and low gene expression corresponds to target label = 0.

Thus, the inputs are 100x5 matrices and target is the probability of gene activity.

## Evaluation
The participants will be ranked based on the accuracy of their predictions. The evaluation metric for this competition is [AUC](https://facebook.github.io/react-native/docs/running-on-device), which assesses the accuracy based on the area under the Receiver Operating Characteristics (ROC) curve. The AUC ranks the submissions based on the order of predicted likelihoods. The desired prediction is likelihood (score between 0 and 1). It is also allowed to submit predicted category (0 or 1), but the score will be less than with well justified likelihoods represented as real numbers.

### Submission Format
Submission files should contain two columns: GeneId and Prediction. The first column of the csv file is the GeneId (1,2,3,...,3870,3871). The second column is the likehood that the genes in testset have a high expression level.

The file should contain a header and have the following format:

```
GeneId, Prediction
1, 0.563237
2, 0.419834
3, 0.959324
. . .
3870, 0.025881
3871, 0.464465
```

## Data Description
The training set consists of two files: x_train.csv and y_train.csv.

x_test.csv is the test set. There is no label information in the testset which the participants need to predict.

### File descriptions
- x_train.csv - It contains six columns separated by comma. The first column is the GeneId. From second till last column, they are the reads in each bin on five core histone modification marks. Each gene has 100 bins.
- y_train.csv - It contains two columns separated by comma. The first column is the GeneId. The second column is its corresponding label (0 or 1) which means this gene has high or low expression level.
- x_test.csv - the test set, similarly formatted as the x_train.csv

### Data fields
- Five core histone modification mask -  H3K4me3, H3K4me1, H3K36me3, H3K9me3, H3K27me3

## References
[1] Kundaje, A. et al. Integrative analysis of 111 reference human epige-
nomes. Nature, 518, 317–330, 2015.

[2] Ritambhara Singh, Jack Lanchantin, Gabriel Robins and Yanjun Qi. ["DeepChrome: deep-learning for predicting gene expression from histone modifications."](https://arxiv.org/pdf/1607.02078.pdf) Journal of Bioinformatics, 32, 1639-1648, 2016.

[Kaggle prediction competition](https://inclass.kaggle.com/c/gene-expression-prediction/data)