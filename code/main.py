# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 21:21:02 2020

@author: Jiang Jilian
"""

from all_models import *
import random
from sklearn.metrics import accuracy_score
from plot_figures import *
from utils.compute_func import *

warnings.filterwarnings("ignore")


def model_train(method, X_train, Y_train, X_pred):
    """
    Trains a model based on the specified method.

    Args:
        method (str): The name of the model to be trained.
        X_train (pd.Series or np.ndarray): Training features.
        Y_train (pd.Series): Training labels.
        _: Placeholder for X_pred; not used here but retained for compatibility.

    Returns:
        The trained model instance.
    """

    if method == 'decisiontree':
        clf = decisiontree(X_train, Y_train)
    if method == 'my_GaussianNB':
        clf = my_GaussianNB(X_train, Y_train)
    if method == 'mlp':
        clf = mlp(X_train, Y_train)
    if method == 'Bernoulli':
        clf = Bernoulli(X_train, Y_train)
    if method == 'randomtree':
        clf = randomtree(X_train, Y_train)
    if method == 'logistic':
        clf = logistic(X_train, Y_train)
    if method == 'SVM':
        clf = SVM(X_train, Y_train)
    if method == 'KNN':
        clf = KNN(X_train, Y_train)
    if method == 'voting':
        clf = voting(X_train, Y_train)
    if method == 'bagging':
        clf = bagging(X_train, Y_train)
    if method == 'adaboost':
        clf = adaboost(X_train, Y_train)
    if method == 'tsvm':
        clf = tsvm(X_train, Y_train, X_pred)

    # Assuming each function returns a trained model
    return clf


if __name__ == '__main__':
    # Set random seed for reproducibility
    random.seed(2)

    # Save experiment logs
    make_print_to_file(output_path=output_path, cv="LeaveOneOut")

    # 1. Load zircon dataset and preprocess labels
    zircons_data = pd.read_excel(data_path + file_name + ".xlsx", header=1)
    zircons_data.loc[zircons_data["Zircon"] == "S-type zircon", "Label"] = 1
    zircons_data.loc[zircons_data["Zircon"] == "I-type zircon", "Label"] = 0
    X_train, Y_train, X_test, Y_test, X_predict, all_X, all_y, processed_data = prepare_data(zircons_data)

    methods = ['decisiontree', 'my_GaussianNB',
               'Bernoulli', 'logistic', 'SVM', 'KNN', 'mlp', 'voting', 'bagging',
               'adaboost', "tsvm"]
    #methods = ["tsvm"]

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # 2. Train machine learning models
    IS_TRAIN = True
    if IS_TRAIN:
        for method in methods:
            print("----------------"+method+"-----------------")
            ## Train the model using training data
            clf = model_train(method, X_train, Y_train, X_predict)

            ## Predict on testing data
            y_test_pred = clf.predict(X_test)
            all_y_pred = clf.predict(all_X)

            # Handle special cases for tsvm and KNN predictions (-1 values)
            if method in ["tsvm", "KNN"]:
                y_test_pred[y_test_pred == -1] = 0
                all_y_pred[all_y_pred == -1] = 0

            ## Compute overall classification report
            processed_data[method + " Label"] = all_y_pred
            classify = classification_report(Y_test, y_test_pred)

            ## Evaluate performance for low-P zircons
            if_low_P = (processed_data["Set"] == "Testing set") & (processed_data["P_raw"] <= 20)
            y_test_pred_low_p = processed_data.loc[if_low_P, method + " Label"]
            classify_low_p = classification_report(processed_data.loc[if_low_P, "Label"], y_test_pred_low_p)

            ## Calculate S-type zircon accuracy for low-P zircons
            if_low_P_S_type = (processed_data["Set"] == "Testing set") & (processed_data["P_raw"] <= 20) & (processed_data["Zircon"] == "S-type zircon")
            y_test_pred_low_p_S_type = processed_data.loc[if_low_P_S_type,  method + " Label"]
            acc_low_P_S_type_zircon = accuracy_score(processed_data.loc[if_low_P_S_type, "Label"], y_test_pred_low_p_S_type)
            classify_low_p = classification_report(processed_data.loc[if_low_P, "Label"], y_test_pred_low_p)


            ## Compute proportions of S-type zircons in JH zircons
            JH_S_zircon_proportion = sum((processed_data['Zircon'] == "JH zircon") & (processed_data[method + ' Label'] == 1)) / sum(processed_data['Zircon'] == "JH zircon")
            JH_Hadean_S_proportion = sum((processed_data['Zircon'] == "JH zircon") & (processed_data[method + ' Label'] == 1) & (
                        zircons_data['Age(Ma)'] >= 4000)) / sum((processed_data['Zircon'] == "JH zircon") & (zircons_data['Age(Ma)'] >= 4000))

            print(classify)
            print(classify_low_p)
            print("Accuracy of Low-P S-type zircon:" + str(acc_low_P_S_type_zircon))
            print("The proportion of S-type zircons in JH zircons: " + str(JH_S_zircon_proportion))
            print("The proportion of S-type zircons in Hadean JH zircons: " + str(JH_Hadean_S_proportion))

            ## Save predicted labels to the dataset
            zircons_data[method + ' Label'] = processed_data[method + ' Label']


            ## Save trained model
            if method == "tsvm":
                clf.save(path=model_path + 'tsvm.model')
            else:
                joblib.dump(clf, model_path + method + ".model")


    # 2. Learning curve and hyperplane function for TSVM
    ## Load model: TSVM
    method = "tsvm"
    cv = LeaveOneOut()
    clf = joblib.load(model_path + method + '.model')

    ## Plot learning curve (omitted due to visualization dependency)
    title = f"Learning Curves for {method}"
    plot_learning_curve(clf, title, X_train, Y_train, ylim=(0.2, 1.1), cv=cv, n_jobs=1)

    ## Perform additional TSVM-specific calculations (e.g., hyperplane function and values)
    hyperplane_function(clf, elements, method)
    svm_y_values = compute_y_value(clf, all_X)
    zircons_data[method + " value"] = pd.DataFrame(svm_y_values, columns=[method + " value"])

    # 3. Update prediction results in database

    zircons_data["Set"] = processed_data['Set']
    for set_type, detrital_label_base in [("Prediction set", "detrital"), ("Training set", ""), ("Testing set", "")]:
        zircons_data.loc[(zircons_data["Set"] == set_type) & (zircons_data[method +' Label'] == 0), method + " model"] = f"I-type {detrital_label_base} zircon"
        zircons_data.loc[(zircons_data["Set"] == set_type) & (zircons_data[method +' Label'] == 1), method + " model"] = f"S-type {detrital_label_base} zircon"

    ## Create a copy of Zircon column for comparison and calculate performance
    train_test_data = zircons_data[zircons_data['Set'] != "Prediction set"].copy()
    train_test_data["Zircon_copy"] = train_test_data["Zircon"].replace({"I-type TTG zircon": "I-type zircon"})
    train_test_data[f'Performance of {method} model'] = np.where(train_test_data['Zircon_copy'] == train_test_data[method + ' model'], 'correct', 'wrong')
    train_test_data.drop("Zircon_copy", axis=1, inplace=True)
    zircons_data[f'Performance of {method} model'] = train_test_data[f'Performance of {method} model']

    ## Save updated dataset with predictions
    zircons_data.to_excel(output_path + file_name + "_with_prediction.xlsx")


    # 3. Predict global detrital zircons and JH zircon, compute the bootstrap of S-type zircon

    global_data = pd.read_excel(data_path + "Global_detrital_zircon_data.xlsx")
    JH_data = zircons_data[zircons_data['Zircon'] == "JH zircon"]
    x_train= zircons_data.loc[(zircons_data["Set"] == "Training set") | (zircons_data["Set"] == "Testing set"), elements]
    global_S_ratio_seq = predict_and_save_time_series(x_train,
                                                        pred_data=global_data,
                                                        method=method,
                                                        output_filename="Bootstrap_means_global_detrital_zircon_data_" + method + ".csv")


    Jh_S_ratio_seq = predict_and_save_time_series(x_train,
                                                     pred_data=JH_data,
                                                     method=method,
                                                     output_filename="Bootstrap_means_JH_zircon_" + method + ".csv")
