import math
import numpy as np
from scipy.stats import stats
from sklearn import preprocessing
import time
import sys
import os
from globalVar import *
import pandas as pd
from sklearn.preprocessing import power_transform
from sklearn.model_selection import train_test_split


def row_ln(row):
    #
    gmean = stats.gmean(np.array(row))
    func = lambda x: math.log(x / gmean, math.e)
    new_row = row.apply(func)
    return new_row


def CLR(x):
    percent_x = (x.T / x.sum(axis=1)).T
    nomalized_x = percent_x.apply(row_ln, axis=1)
    return nomalized_x


def prepare_data(zircons_data):
    """
    Prepares the data by splitting it into training, testing, and prediction sets,
    normalizing the feature data, and returning relevant subsets.

    Args:
        zircons_data (pd.DataFrame): The original dataset with labels and features.

    Returns:
        tuple: A tuple containing (X_train, Y_train, X_test, Y_test, X_predict, all_X, all_y, data)
            where:
                - X_train: Training set features.
                - Y_train: Training set labels.
                - X_test: Testing set features.
                - Y_test: Testing set labels.
                - X_predict: Prediction set features.
                - all_X: All normalized feature data.
                - all_y: All labels.
                - data: Processed dataframe with new 'Set' column indicating train/test/predict.
    """
    # Drop rows with missing values in elements
    zircons_data.dropna(subset=elements, inplace=True)
    zircons_data.reset_index(drop=True, inplace=True)

    # Split raw data into training, testing, and prediction sets
    train_mask = zircons_data["Set"] == "Training set"
    test_mask = zircons_data["Set"] == "Testing set"
    pred_mask = zircons_data["Set"] == "Prediction set"

    # Randomly split training and testing sets maintaining class balance
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        zircons_data.loc[train_mask | test_mask, elements], zircons_data.loc[train_mask | test_mask,"Label"],
        stratify=zircons_data.loc[train_mask | test_mask, "Label"],
        test_size=0.2, shuffle=True, random_state=19)

    # Create new datasets with appended labels
    new_train_set = pd.concat([x_train_raw, pd.DataFrame(y_train, columns=["Label"])], axis=1)
    new_test_set = pd.concat([x_test_raw, pd.DataFrame(y_test, columns=["Label"])], axis=1)

    # prediction set
    raw_prediction_set = zircons_data[pred_mask]
    raw_prediction_set.reset_index(inplace=True, drop=True)

    group_data_train = new_train_set.groupby(['Label'])['Label'].count()
    # check out the sample number of testing set
    print("Training set: ")
    print(group_data_train)

    group_data_test = new_test_set.groupby(['Label'])['Label'].count()
    # check out the sample number of training set
    print("Testing set: ")
    print(group_data_test)

    # Normalize all zircons
    x_data = preprocess_data(x_train_raw, zircons_data[elements], normalize_method="CLR")


    # Update dataset with normalized features and new 'Set' column
    zircons_data["P_raw"] = zircons_data["P (μmol/g)"].copy()
    data = pd.concat([pd.DataFrame(x_data, columns=elements), zircons_data[info_list + ["Label", "P_raw"]]], axis=1)
    data.loc[y_train.index, ["Set"]] = "Training set"
    data.loc[y_test.index, ["Set"]] = "Testing set"

    # Re-assign processed sets
    train_set = data[(data["Set"] == "Training set")]
    train_set.reset_index(inplace=True, drop=True)
    test_set = data[(data["Set"] == "Testing set")]
    test_set.reset_index(inplace=True, drop=True)
    predict_set = data[(data["Set"] == "Prediction set")]
    predict_set.reset_index(inplace=True, drop=True)

    print("--------------------------------")

    # Prepare final outputs
    all_X = x_data
    all_y = zircons_data["Label"]
    X_train = train_set[elements]
    Y_train = train_set["Label"]
    X_test = test_set[elements]
    Y_test = test_set["Label"]
    X_predict = predict_set[elements]

    return X_train, Y_train, X_test, Y_test, X_predict, all_X, all_y, data


def preprocess_data(x_train, x, normalize_method="CLR", scaling=True):
    """
    Transform the dataset into standard uniform
    """
    if normalize_method == "CLR":
        pro_x = CLR(x)
        pro_x_train = CLR(x_train)
    elif normalize_method == "BOX-COX":
        pro_x = power_transform(x, method='box-cox')
        pro_x_train = power_transform(x_train, method='box-cox')
    elif normalize_method == "Z-score":
        pro_x = x
        pro_x_train = x_train

    if scaling:
        scaler = preprocessing.StandardScaler().fit(pro_x_train)
        X = scaler.transform(pro_x)
    else:
        X = pro_x

    return X


def make_print_to_file(output_path='./', cv=0):
    """
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    """

    class Logger(object):
        def __init__(self, filename="Default.log", stream=sys.stdout):
            self.terminal = stream
            # self.terminal = sys.stdout
            self.log = open(filename, "a")
            # encoding='utf8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    timetamp = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    sys.stdout = Logger(output_path + file_name + "_evaluation_" + str(cv) + "_"+ timetamp + '.log', sys.stdout)

    print("The fold of cross validation: " + str(cv) + ".")
    print(file_name.center(60, '*'))
    # print("______---------------------------------")
    # return fileName
