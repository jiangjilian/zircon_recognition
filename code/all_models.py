# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 00:29:44 2021

@author: 郑宗源
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier

import os
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomTreesEmbedding, ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier

from pylab import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.ticker as plticker
from globalVar import *
from sklearn.svm import SVC
import numpy as np

np.random.seed(1337)  # for reproducibility


def decisiontree(X_train, y_train):
    # decision_tree
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    param = {'criterion': ['gini', 'entropy'],
             'max_depth': [5, 10, 20, 40],
             'min_samples_leaf': [2, 3, 5, 10],
             'min_samples_split': [2, 4, 6, 8, 10]
             }
    grid = RandomizedSearchCV(clf, param, cv=LeaveOneOut(), scoring='f1', return_train_score=False)
    grid.fit(X_train, y_train)
    print("Best param:", grid.best_params_)

    clf = grid

    return clf


def my_GaussianNB(X_train, y_train):
    """
    Guassian Naive Bayes
    """
    std_clf = GaussianNB()
    std_clf.fit(X_train, y_train)
    eclf = CalibratedClassifierCV(std_clf, method="isotonic", cv="prefit")
    eclf.fit(X_train, y_train)

    return eclf


def mlp(X_train, y_train):
    """
    Multilayer Perceptron
    """

    clf = MLPClassifier()
    param = {'hidden_layer_sizes': [(100,), (100, 40), (100, 80, 40)],
             'solver': ['adam', 'sgd', 'lbfgs'],
             'max_iter': [100, 1000],
             'learning_rate_init': [0.001, 0.01, 0.05, 0.1],
             # 'early_stopping': [True]
             }

    grid = RandomizedSearchCV(clf, param, cv=LeaveOneOut(), scoring='f1', return_train_score=False)
    grid.fit(X_train, y_train)
    print("Best param:", grid.best_params_)

    clf = grid

    return grid


def Bernoulli(X_train, y_train):
    """
    Bernoulli

    """
    clf = BernoulliNB()
    param = {'alpha':  [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0],
             'fit_prior': [True],
             #'class_prior': [2]
             }

    grid = RandomizedSearchCV(clf, param, cv=LeaveOneOut(), scoring='f1')
    grid.fit(X_train, y_train)
    print("Best param:", grid.best_params_)

    clf = grid

    return clf


def randomtree(X_train, y_train):
    clf = RandomForestClassifier()
    param = {'criterion': ['gini', 'entropy'],
             'n_estimators': [10, 100, 200, 400, 1000],
             'max_depth': [5, 10, 20, 40],
             'min_samples_leaf': [2, 3, 5, 10],
             'min_samples_split': [2, 4, 6, 8, 10]
             }

    grid = RandomizedSearchCV(clf, param, cv=LeaveOneOut(), scoring='f1')
    #, bootstrap=True, oob_score=True
    grid.fit(X_train, y_train)
    print("Best param:", grid.best_params_)

    clf = grid

    return clf


def SVM(X_train, y_train):
    params = {'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
              'C': [0.001, 0.01, 0.1,0.2,0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 5.0, 10.0],
              'gamma': [0.001, 0.01, 0.1, 1, 2, 4, 8, 10, 100, 1000]
              }
    svc = svm.SVC()
    cv = LeaveOneOut()
    grid = RandomizedSearchCV(svc, params, cv=cv, scoring='f1')
    grid.fit(X_train, y_train)
    print("Best param:", grid.best_params_)

    best_params = grid.best_params_
    clf = svm.SVC(**best_params, tol=0.0005)
    clf.fit(X_train, y_train)

    return clf


def KNN(X_train, y_train):
    params = {'n_neighbors': [1, 2, 3, 4, 5, 8, 10, 15, 20, 50, 100],
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
              }
    clf = KNeighborsClassifier()
    grid = RandomizedSearchCV(clf, params, cv=LeaveOneOut(), scoring='f1')
    grid.fit(X_train, y_train)
    print("Best param:", grid.best_params_)

    best_params = grid.best_params_
    clf = KNeighborsClassifier(**best_params)

    clf.fit(X_train, y_train)

    return clf


def logistic(X_train, y_train):
    clf = LogisticRegression()
    param = {'penalty': ['l1', 'l2'],
             'C': [0.01, 0.05, 0.1, 1, 2, 4, 8, 10, 100],
             'solver': ['lbfgs', 'sag', 'saga'],
             'multi_class': ['multinomial']
             }

    grid = RandomizedSearchCV(clf, param, cv=LeaveOneOut(), scoring='f1')
    # , bootstrap=True, oob_score=True
    grid.fit(X_train, y_train)
    print("Best param:", grid.best_params_)
    clf = grid

    return clf


def voting(X_train, y_train):
    clf1 = LogisticRegression(random_state=1, max_iter=2000)
    clf2 = RandomForestClassifier(random_state=1)

    # priors = 3
    # priors=priors
    clf3 = GaussianNB()
    eclf = VotingClassifier(
        estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
        voting='soft')

    dtc = DecisionTreeClassifier(random_state=1)
    rfc = RandomForestClassifier(random_state=1)
    ada = AdaBoostClassifier(random_state=1)
    gdb = GradientBoostingClassifier(random_state=1)
    eclf = VotingClassifier(estimators=[('dtc', dtc), ('rfc', rfc),
                                        ('ada', ada), ('gdb', gdb)], voting='soft')

    eclf.fit(X_train, y_train)

    return eclf


def bagging(X_train, y_train):
    # bagging
    tree = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=None)
    clf = BaggingClassifier(base_estimator=tree,
                            bootstrap_features=False,
                            n_jobs=-1,
                            random_state=1)

    param = {'n_estimators': [10, 100, 200],
             'max_samples': [0.8, 0.9, 1.0],
             'max_features': [0.8, 0.9, 1.0]
             }

    grid = GridSearchCV(clf, param_grid=param, cv=LeaveOneOut(), scoring='f1')
    #, bootstrap=True, oob_score=True
    grid.fit(X_train, y_train)
    print("Best param:", grid.best_params_)
    clf = grid

    return clf



def adaboost(X_train, y_train):
    clf = AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators=1000)
    param = {'n_estimators': [10, 100, 200, 400],
             'learning_rate': [0.1, 0.5, 0.8, 1.0, 2.0],
             }

    grid = GridSearchCV(clf, param_grid=param, cv=LeaveOneOut(), scoring='f1')
    #, bootstrap=True, oob_score=True
    grid.fit(X_train, y_train)
    print("Best param:", grid.best_params_)
    clf = grid

    return clf


def Vote2(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf1 = make_pipeline(MinMaxScaler(), MLPClassifier(max_iter=1000000))
    clf2 = ExtraTreesClassifier()
    clf3 = make_pipeline(GaussianNB())
    # PCA(),
    clf4 = make_pipeline(svm.SVC(probability=True))
    # StandardScaler(),
    eclf = VotingClassifier(
        estimators=[('mlp', clf1), ('rf', clf2), ('gnb', clf3), ('svm', clf4)],
        voting='soft').fit(X_train, y_train)
    eclf.fit(X_train, y_train)

    return eclf
"""
class TSVM:
    def __init__(self, kernel='linear', C=1.0, gamma='auto'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.clf = svm.SVC(C=C, kernel=self.kernel, random_state=0)
        #self.nu = nu

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        # Step 1: Train an SVM with labeled data and get support vectors
        svm_labeled = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        svm_labeled.fit(X_labeled, y_labeled)
        support_vector_indices = svm_labeled.support_ # indices of support vectors
        support_vectors = svm_labeled.support_vectors_ # support vectors' features
        support_vector_labels = svm_labeled._get_coef()[0] # support vectors' labels
        positive_sv_indices = np.where(support_vector_labels > 0)[0] # indices of positive class support vectors
        negative_sv_indices = np.where(support_vector_labels < 0)[0] # indices of negative class support vectors

        # Step 2: Get "virtual" labels for unlabeled data and retrain the model
        pseudo_labeled = np.zeros(X_unlabeled.shape[0])
        pseudo_labeled[np.ravel((np.dot(np.array(X_unlabeled), svm_labeled.coef_.T) + svm_labeled.intercept_) > 0)] = 1 # labeling positive examples
        pseudo_labeled[np.ravel((np.dot(np.array(X_unlabeled), svm_labeled.coef_.T) + svm_labeled.intercept_) < 0)] = -1 # labeling negative examples

        # Step 3: Retrain the SVM with both labeled and virtual-labeled data
        X_combined = np.vstack((X_labeled, X_unlabeled))
        y_combined = np.hstack((y_labeled, pseudo_labeled))
        svm_combined = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        svm_combined.fit(X_combined, y_combined)

        # Step 4: Select new positive and negative support vectors
        dual_coef = svm_combined._get_coef()[0]
        positive_sv_indices_new = []
        negative_sv_indices_new = []
        for i in range(len(dual_coef)):
            # if it is a positive class support vector and its corresponding Lagrange multiplier > 0, add it to the list of positive support vectors
            if i in positive_sv_indices and dual_coef[i] > 0:
                positive_sv_indices_new.append(i)

            # if it is a negative class support vector and its corresponding Lagrange multiplier < 0, add it to the list of negative support vectors
            if i in negative_sv_indices and dual_coef[i] < 0:
                negative_sv_indices_new.append(i)

        # Step 5: Retrain the SVM with new support vectors and return the model
        support_vector_indices_new = support_vector_indices[np.hstack((positive_sv_indices_new, negative_sv_indices_new))]
        support_vectors_new = support_vectors[np.hstack((positive_sv_indices_new, negative_sv_indices_new))]
        support_vector_labels_new = support_vector_labels[np.hstack((positive_sv_indices_new, negative_sv_indices_new))]
        svm_final = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        svm_final.support_ = support_vector_indices_new
        svm_final.support_vectors_ = support_vectors_new
        svm_final._get_coef = lambda: ([1] if len(positive_sv_indices_new) + len(negative_sv_indices_new) == 0 else support_vector_labels_new)
        self.clf = svm_final

    def predict(self, X):
        return self.clf.predict(X)
"""

class TSVM(SVC):
    def __init__(self, kernel="linear", C=1.0, Cl=1.0):
        self.kernel = kernel
        self.dataset = None
        # self.maxiter = maxiter
        # maxiter=500, tol=1e-6
        # self.tol = tol
        # self.funvalue = None
        # self.coef = None
        #self.iteration = maxiter
        self.C, self.Cl, self.Cu = C, Cl, 0.001
        self.clf = svm.SVC(C=C, kernel=self.kernel, random_state=0)

    def fit(self, X1, Y1, X2):
        """
        Train TSVM by X1, Y1, X2
        Parameters
        ----------
        X1: Input data with labels
                np.array, shape:[n1, m], n1: numbers of samples with labels, m: numbers of features
        Y1: labels of X1
                np.array, shape:[n1, ], n1: numbers of samples with labels
        X2: Input data without labels
                np.array, shape:[n2, m], n2: numbers of samples without labels, m: numbers of features
        """
        N = len(X1) + len(X2)
        # print(N)
        sample_weight = np.ones(N)
        sample_weight[len(X1):] = self.Cu

        self.clf.fit(X1, Y1)
        Y2 = self.clf.predict(X2)
        Y1 = np.expand_dims(Y1, 1)
        Y2 = np.expand_dims(Y2, 1)
        X2_id = np.arange(len(X2))
        X3 = np.vstack([X1, X2])
        Y3 = np.vstack([Y1, Y2])

        while self.Cu < self.Cl:
            self.clf.fit(X3, Y3, sample_weight=sample_weight)
            while True:
                Y2_d = self.clf.decision_function(X2)  # linear: w^Tx + b
                Y2 = Y2.reshape(-1)
                epsilon = 1 - Y2 * Y2_d  # calculate function margin
                positive_set, positive_id = epsilon[Y2 > 0], X2_id[Y2 > 0]
                negative_set, negative_id = epsilon[Y2 < 0], X2_id[Y2 < 0]
                positive_max_id = positive_id[np.argmax(positive_set)]
                negative_max_id = negative_id[np.argmax(negative_set)]
                a, b = epsilon[positive_max_id], epsilon[negative_max_id]
                if a > 0 and b > 0 and a + b > 2.0:
                    Y2[positive_max_id] = Y2[positive_max_id] * -1
                    Y2[negative_max_id] = Y2[negative_max_id] * -1
                    Y2 = np.expand_dims(Y2, 1)
                    Y3 = np.vstack([Y1, Y2])
                    self.clf.fit(X3, Y3, sample_weight=sample_weight)
                else:
                    break
            self.Cu = min(2 * self.Cu, self.Cl)
            sample_weight[len(X1):] = self.Cu
        # return self

    def predict(self, X):
        """
        Feed X and predict Y by TSVM
        """
        return self.clf.predict(X)

    def score(self, X, Y):
        """
        Calculate accuracy of TSVM by X, Y
        Parameters
        ----------
        X: Input data
                np.array, shape:[n, m], n: numbers of samples, m: numbers of features
        Y: labels of X
                np.array, shape:[n, ], n: numbers of samples
        Returns
        -------
        Accuracy of TSVM
                float
        """
        return self.clf.score(X, Y)

    def predict_proba(self, X):
        """
        Feed X and predict Y by TSVM
        """

    def save(self, path='./TSVM.model'):
        """
        Save TSVM to model_path
        """
        joblib.dump(self.clf, path)

    def decision_function(self, X):
        """
        Feed X and predict Y by TSVM
        """
        return self.clf.decision_function(X)

    def classification_report(self, X):
        return self.clf.classification_report(X)


def tsvm(X_train, y_train, X_unlabeled):
    # 定义参数搜索空间
    #param_grid = {'svc__C': [0.1, 0.5,  1, 5, 10], 'svc__kernel': ['rbf', 'linear'], 'tsvm__gamma': [0.1, 1, 10, "auto"], 'nu': [0.1, 0.3, 0.6, 0.9]}
    """
    cv = LeaveOneOut()
    tsvm = TSVM()
    
    grid = RandomizedSearchCV(tsvm, param_grid, cv=cv, scoring='f1')
    grid.fit(X_train, y_train, X_unlabeled)
    print("Best param:", grid.best_params_)
    """
    # 预测未标记数据并计算准确率
    # best_params = grid.best_params_
    y_train[y_train == 0] = -1
    # y_val[y_val == 0] = -1
    # Y_test[Y_test == 0] = -1
    clf = TSVM(kernel='linear', C=3.0, Cl=0.5)
    clf.fit(X_train, y_train, X_unlabeled)

    return clf


def tune_tsvm_parameters(tuned_parameters, X_train, Y_train, X_test, Y_test, X_pred, cv):
    Y_train[Y_train == 0] = -1
    # y_val[y_val == 0] = -1
    Y_test[Y_test == 0] = -1
    # y_our_test[y_our_test == 0] = -1
    #X_train, Y_train = shuffle(X_train, Y_train)
    best_validation_score = 0
    best_parameters = {}
    validation_score_df = pd.DataFrame()
    i = 0
    for kernel_i in tuned_parameters["kernel"]:
        for C in tuned_parameters["C"]:
            for Cl in tuned_parameters["Cl"]:
                clf = TSVM(kernel=kernel_i, C=C, Cl=Cl)
                # clf = TSVM()
                # clf.initial(Cl, C)
                pred_test_x = pd.concat([X_pred, X_test], axis=0)
                #pred_test_x = X_pred.copy()
                pred_test_x.reset_index(inplace=True, drop=True)
                y_val_pred_list = []
                y_val_list = []
                for train_index, validation_index in cv.split(X_train):
                    x_train = X_train.loc[train_index, :]
                    y_train = Y_train[train_index]
                    x_val = X_train.loc[validation_index, :]
                    y_val = Y_train[validation_index]
                    clf.fit(x_train, y_train, pred_test_x)
                    # y_train_pred = clf.predict(X_train)
                    # y_val_pred = clf.predict(x)

                    # pred_y_test = clf.predict(X_test)
                    y_val_pred = clf.predict(x_val)
                    y_val_list.append(y_val)
                    y_val_pred_list.append(y_val_pred)
                    # validation_score = accuracy_score(y_val, y_val_pred)

                # validation_score = accuracy_score(y_val_list, y_val_pred_list)
                validation_score = f1_score(y_val_list, y_val_pred_list, average='macro')
                # , average='micro'
                # validation_score_macro = f1_score(y_val, y_val_pred, average='macro')
                #p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_true, y_pred,
                #                                                                           labels=[1, 2, 3])

                # test_data["pred_type"] = pred_y_test
                validation_score_df.loc[i, "validation_score"] = validation_score
                validation_score_df.loc[i, "C"] = C
                validation_score_df.loc[i, "Cl"] = Cl
                i = i + 1
                if validation_score > best_validation_score:  # 找到表现最好的参数
                    best_validation_score = validation_score
                    best_parameters = {'kernel': kernel_i, 'C': C, 'Cl': Cl}

                    # clf.fit(X_train, Y_train, pred_test_x)
                    # test_score = clf.score(X_test, Y_test)
                    # test_score = accuracy_score(Y_test, clf.predict(X_test))
                    # joblib.dump(clf, './best_' + str(tuned_parameters["kernel"]) + '_TSVM.model')
    ####   grid search end
    validation_score_df.to_csv(dataPath + "TSVM_tune_parameters_validation_score.csv")
    print("Best validation score:{:.2f}".format(best_validation_score))
    # print("Test accuracy:{:.2f}".format(test_score))
    print("Best parameters:{}".format(best_parameters))





