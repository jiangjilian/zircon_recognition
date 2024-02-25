import pandas as pd
import numpy as np
from sklearn.utils import resample
#from normlization import *
import joblib
from utils.normlization import *


def predict_and_save_time_series(x_train, pred_data, method, output_filename):
    """
    Load global detrital zircon data, preprocess it, make predictions using a pre-trained model,
    and save the computed time series statistics to a CSV file.

    Args:
    method (str): The machine learning method used in the model (e.g., 'SVM', 'KNN').

    """
    # Step 12: Preprocess data
    pred_data_x = preprocess_data(x_train, pred_data[elements], normalize_method="CLR")

    # Step 2: Load model
    clf = joblib.load(f"{model_path}/{method}.model")

    # Step 3: Make predictions
    pred_labels = clf.predict(pred_data_x)

    # Step 4: Adjust negative labels if applicable
    if method.lower() in ["tsvm", "knn"]:
        pred_labels[pred_labels == -1] = 0

    # Step 5: Add predicted labels to DataFrame
    pred_data["Label"] = pred_labels

    # Step 6: Prepare plot data
    plot_data = pred_data[["Age(Ma)", "Label"]]
    col_age = "Age(Ma)"
    col_label = "Label"

    # Step 7: Compute time series stats
    s_ratio_sequence = compute_time_seq(plot_data, col_age, col_label, series_type="S ratio")

    # Step 8: Save time series stats to CSV
    s_ratio_sequence.to_csv(f"{output_path}/{output_filename}")


def compute_time_seq(data, x_col='Age(Ma)', y_col='Label', series_type="S ratio"):
    """
    Compute the time series of a specific type of zircon ratio along with its statistical measures.

    Parameters:
    data (pandas.DataFrame): DataFrame containing 'Age（Ma)' and 'Label' columns
    x_column_name (str): Name of the age column; default is 'Age（Ma)'
    y_column_name (str): Name of the label column; default is 'Label'
    series_type (str): Type of zircon ratio to be computed; default is "S ratio"

    Returns:
    result (pandas.DataFrame): A DataFrame summarizing the statistics of zircon type ratios for each time bin

    Note: This function divides the data into bins based on time intervals, calculates the S-type zircon ratio using bootstrapping when there are at least 4 samples in a bin,
          and computes mean, standard deviation, and standard error. If fewer than 4 samples exist in a bin, no statistical calculations are performed.
    """
    AGE = data[x_col]
    Element_data = data[y_col]

    sampleN = int(len(AGE))

    X1 = 4500
    X2 = 4600
    step = 100
    X_limited = 0

    low = X1
    high = X2
    result = pd.DataFrame(data=None)

    nA = [np.nan] * int((X1 - X_limited) / step + 2)
    S_num = []

    for j in np.arange(0, int((X1 - X_limited) / step + 2), 1):
        # dataAA=[]
        BinAA = Element_data.copy()
        BinAA[BinAA[(AGE < low) | (AGE > high)].index] = np.nan
        dataAA = BinAA[BinAA[~np.isnan(BinAA)].index]
        nA[j] = len(dataAA)
        S_num.append(sum(dataAA))
        # print(nA[j])
        result.loc[j, "AGE_MEDIAN"] = (low + high) / 2  # age

        if nA[j] >= 4:  # less than 4 samples will not be calculated.
            iter = 1000
            S_ratio_list = []
            for i in range(iter):
                bootstrapSamples = resample(dataAA, n_samples=100, replace=1)
                temple_S_ratio = scale_S_ratio(bootstrapSamples)  # single S_ratio
                S_ratio_list.append(temple_S_ratio)

            # CIs = bootstrp.ci(data=dataAA, statfunction=sp.mean, n_samples=10000)
            result.loc[j, str(series_type) + " mean"] = np.mean(S_ratio_list)
            result.loc[j, str(series_type) + " std"] = np.std(S_ratio_list)
            result.loc[j, str(series_type) + " sem"] = np.std(S_ratio_list) / np.sqrt(len(dataAA))  # standard error

        else:
            result.loc[j, str(series_type) + " mean"] = np.nan
            result.loc[j, str(series_type) + " std"] = np.nan  # standard error

        result.loc[j, "total num"] = nA[j]
        result.loc[j, str(series_type) + " num"] = S_num[j]
        low = low - step  # define the bin size (step width)
        high = high - step  # define the bin size (step width)

    return result


def scale_S_ratio(samples):
    count = 0.0
    total = samples.size
    for i in samples:
        if (i == 1):
            count += 1.0
    return count / (total)


def compute_y_value(model, x):
    """
    Computes the output value(s) from the model's hyperplane given input data.

    Args:
        model: A trained model.
        x (np.ndarray): Input feature vector.

    Returns:
        np.ndarray: Array of predicted values based on the model's hyperplane.
    """
    w = model.coef_[0]
    b = model.intercept_[0]
    return np.dot(w, x.T) + b


def hyperplane_function(model, elements, modelName):
    """
    Prints the coefficients, intercept, and the equation of the hyperplane.

    Args:
        model: A trained model.
        elements (list[str]): List of feature names.
        modelName (str): Name of the model for printing purposes.
    """
    w = model.coef_[0]
    b = model.intercept_[0]

    print(f"{modelName} coefficients:")
    print(w)
    print(f"{modelName} intercept:")
    print(b)

    function_text = " + ".join([f"{w[i]} * {e}" for i, e in enumerate(elements)]) + f" + {b} = 0"
    print(f"{modelName} hyperplane function:")
    print(function_text)


