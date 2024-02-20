import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# This file will be Modelling and Analysis
# Data Split, Feature Selection, Data Test, Precision and Recall
 
# Selecting features with has the most correlation to classification of protein
# We will be using Mutual Information with each column against the class_labels, which is the first two digits of the Classification compile
# The features and their correlation will be stored in a array of tuples
def feature_correlation(featurenames,class_label, random=100):
    # Feature selection through correlation between nutrients and protein class 
    mi_features = []
    mi_arr = mutual_info_classif(X=featurenames, y=class_label, random_state=random)

    for feature, mi in zip(featurenames.columns, mi_arr):
        mi_features.append((feature, mi))

    return mi_features

# Correlations between features
def feature_reduction(df, threshold, random = 100):
    # Features with high correlation will be filtered out of the dataset
    # Correlation between every pair of features will be conducted
    similar = []
    for feature1 in range(len(df.columns)):
        for feature2 in range(feature1 + 1, len(df.columns)):
            """
            A function which computes the Pearson Correlation between two features
            """
            if df.columns[feature2] not in similar:
                feature_a = df[(df.columns[feature1])]
                feature_b = df[(df.columns[feature2])]
                # compute the mean
                mean_a = feature_a.mean()
                mean_b = feature_b.mean()
                
                # compute the numerator of pearson r
                numerator = sum((feature_a - mean_a) * (feature_b - mean_b))
                
                # compute the denominator of pearson r
                denominator = np.sqrt(sum((feature_a - mean_a) ** 2) * sum((feature_b - mean_b) ** 2))
                
                correlation = numerator/denominator

                if abs(correlation > threshold):
                    similar.append(df.columns[feature2])
    return df[similar]


# Strip columns with correlation below a threshold
def strippedfeatures(threshold, correlation):
    # Returns a list of features above threshold ValueError
    answer = []
    for feature in correlation:
        if feature[1] >= threshold:
            answer.append(feature[0])
    return answer

## Classification Score Evaluater
def scorePriority(report):
    total = 0
    for class_label in report:
        if type(report[class_label]) != dict:
            break
        total += report[class_label]['support']
    
    weights = []
    for class_label in report:
        if type(report[class_label]) != dict:
            break
        inverted = total/(report[class_label]['support'])
        weights.append(inverted)

    final_score = 0
    count = 0
    for class_label in report:
        if type(report[class_label]) != dict:
            break
        score = weights[count]/sum(weights)
        count+= 1
        final_score += score*report[class_label]['f1-score']
    return final_score


## Training and testing dataset with selected features
def trainAndTest(features, df, class_label, random =100, nfolds = 3):
    
    # We will conducting Classification machine learning on a selected features
    # First we filter out columns not in features
    X = df[features]
    X = np.asmatrix(X)
    y = class_label
    # For data splitting, we will be using Cross-Validation to reduce biases and prevent overfitting of the data
    # Cross-Validation will seperate the dataset into n-folds and training and testing will be conducted n times 
    # where each fold will be test and the remaining will be used for training for each run

    nf_CV = KFold(n_splits=nfolds, shuffle=True, random_state=random)
    report_test = []
    report_pred = []
    for train_idx, test_idx in nf_CV.split(X):
        # train-test split
        X_train, X_test = np.asarray(X[train_idx]), np.asarray(X[test_idx])
        y_train, y_test = np.asarray(y[train_idx]), np.asarray(y[test_idx])
        # Preprocessing
        # 1. Standardise the data
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Training
        knn = KNN(n_neighbors=5)
        knn.fit(X_train, y_train)    
        
        # Predictions
        y_pred = knn.predict(X_test)
        report_test = np.concatenate((report_test,y_test), axis = None)
        report_pred = np.concatenate((report_pred,y_pred), axis = None)

    """Support is the number of actual occurrences of the class in the specified dataset. 
    Imbalanced support in the training data may indicate structural weaknesses in the reported scores of the classifier and could indicate the need for stratified sampling or rebalancing. 
    Support doesn’t change between models but instead diagnoses the evaluation process."""
    return (report_test, report_pred)

# Classifying the unclassified excel sheet
def Test(features, df, class_label, unclassified, random =100, nfolds = 3):

    X = df[features]
    X = np.asmatrix(X)
    y = class_label
    
    test = np.asarray(unclassified[features])
    nf_CV = KFold(n_splits=nfolds, shuffle=True, random_state=random)
    report_test = []
    report_pred = []
    for train_idx, test_idx in nf_CV.split(X):
        # train-test split
        X_train, X_test = np.asarray(X[train_idx]), np.asarray(X[test_idx])
        y_train, y_test = np.asarray(y[train_idx]), np.asarray(y[test_idx])
        # Preprocessing
        # 1. Standardise the data
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Training
        knn = KNN(n_neighbors=5)
        knn.fit(X_train, y_train)    
        
        # Predictions
        y_pred = knn.predict(X_test)

    return knn.predict(test)

def selected_threshold(correlation_scores, features, class_label, random = 100):
    # Finding the best threshold
    best_threshold = (0,0,0) # Accuracy, WeightedScore, Threshold

    # Iterate through every every threshold value from 1 - 100%
    for threshold in range(1,100):
        selected_features = strippedfeatures((threshold/100), correlation_scores)
        if len(selected_features) > 0:
            result = trainAndTest(selected_features, features, class_label, random)
            report = classification_report(result[0], result[1], output_dict=True)
            accuracy = accuracy_score(result[0],result[1])
            weightedscore = scorePriority(report)

            # If score is better or same value, we update because there is less features
            if weightedscore >= best_threshold[1]:
                if accuracy >= best_threshold[0]:
                    best_threshold = accuracy, weightedscore, threshold
                    cm = confusion_matrix(result[0], result[1])

    count = 0
    cm2 =[[],[],[],[],[]]
    for class_label in report:
        if type(report[class_label]) != dict:
            break

            ## FIX THIS
        for x in range(len(cm)):
            cm2[x].append(100*cm[x][count]/report[class_label]['support'])
        count += 1
    fig = plt.figure()
    plt.matshow(cm2)
    plt.title('Confusion Matrix for Classification')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig('confusion_matrix.jpg')
    return best_threshold

