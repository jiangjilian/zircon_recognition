#!/usr/bin/env python
# coding: utf-8
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from globalVar import *
from utils.normlization import *
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.ticker import MultipleLocator
from sklearn.model_selection import learning_curve
warnings.filterwarnings("ignore")

from matplotlib import rcParams

config = {
    "font.family": 'Arial',
    "axes.unicode_minus": False,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}
rcParams.update(config)
def plot_fig1A(input_file, output_file):
    """
    This function plots (REE+Y)3 versus Y relationship for different zircon types.

    Parameters:
    input_file (str): The path to the Excel file containing data.
    output_file (str): The base name of the output files. The figures will be saved as .jpg and .pdf formats.

    Description:
    1. Reads in data from a specific sheet within an Excel file.
    2. Categorizes 'Zircon' type based on SVM model predictions.
    3. Plots (REE+Y)3 against Y for each zircon type with distinct colors and markers.
    4. Adds dashed lines for reference and sets axis limits.
    5. Saves the figure in both JPEG and PDF formats.

    Zircon Types and their associated colors:
    - Detrital zircon: Face color - orange, Edge color - none
    - I-type zircon: Face color - white, Edge color - red
    - S-type zircon: Face color - white, Edge color - steelblue

    Note: The legend handles are resized for better visibility.

    """

    # Define zircon types and corresponding colors & facecolors
    zircon_types = [
        'Detrital zircon',
        'I-type zircon',
        'S-type zircon',
    ]
    edge_colors = [
        'none',
        'red',
        'steelblue',
    ]
    face_colors = [
        'orange',
        'white',
        'white',
    ]
    transparency = [0.3, 1.0, 1.0]  # Transparency levels for each zircon type

    # Read data from the Excel file, assuming the "Zircon" classification is in column "SVM model"
    df_JH_plot = pd.read_excel(input_file)
    df_JH_plot.loc[df_JH_plot['Set'] == "Prediction set", "Zircon"] = "Detrital zircon"

    # Initialize the figure and axes
    fig, ax = plt.subplots(figsize=(4, 3))

    # Scatter plot for each zircon type
    for zircon_type, edge_color, face_color, alpha in zip(zircon_types, edge_colors, face_colors, transparency):
        scatter_data = df_JH_plot[df_JH_plot["Zircon"] == zircon_type]
        sc = plt.scatter(
            scatter_data["P (μmol/g)"],
            scatter_data["REE+Y (μmol/g)"],
            s=15.0,
            facecolors=face_color,
            edgecolors=edge_color,
            linewidth=1.0,
            alpha=alpha,
            label=zircon_type,
        )

    # Customize legend
    lgnd = plt.legend(fontsize=6, loc="lower right")
    for handle in lgnd.legendHandles:
        handle.set_sizes([6])

    # Set plot boundaries and add guidelines
    plt.xlim(0, 70)
    plt.ylim(0, 70)
    plt.plot([0, 70], [0, 70], color="k", lw=0.5, linestyle="-")  # Diagonal line for 1:1 ratio
    plt.plot([0, 37], [0, 70], linestyle="dashed", color="k", lw=0.5)  # Dashed guideline
    plt.plot([0, 70], [0, 28], linestyle="dashed", color="k", lw=0.5)  # Another dashed guideline

    # Set tick locations and labels
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=10))
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=10))
    ax.tick_params(axis="y", direction="in", labelsize=8)
    ax.tick_params(axis="x", direction="in", labelsize=8)

    # Add axis labels
    plt.xlabel('P (μmol/g)', fontsize=8)
    plt.ylabel('(REE+Y)3+ (μmol/g)', fontsize=8)

    # Save the figure and show it
    plt.tight_layout()
    plt.savefig(f"{output_file}.jpg", dpi=300)
    plt.savefig(f"{output_file}.pdf")
    plt.show()


def plot_fig1B(input_file, output_file):
    """
    This function plots the age (in billions of years) against P concentration (μmol/g)
    for different types of zircon samples.

    Parameters:
    input_file (str): The path to the Excel file containing data.
    output_file (str): The base name of the output files. Figures will be saved as .jpg and .pdf formats.

    Description:
    1. Reads in data from an Excel file.
    2. Plots ages versus P concentration using scatter points with distinct markers, colors, and facecolors.
    3. Customizes the legend, axis limits, tick locations, and labels.
    4. Saves the figure in both JPEG and PDF formats.

    Zircon Types and their associated properties:
    - Detrital zircon: Marker - '.', Face color - gray, Edge color - none
    - JH zircon: Marker - '.', Face color - orange, Edge color - none
    - S-type zircon: Marker - 'o', Face color - none, Edge color - steelblue
    - I-type zircon: Marker - 'o', Face color - none, Edge color - red
    - Archean detrital zircon (commented out): Marker - '.', Face color - gray, Edge color - none

    Note: Sizes of the legend handles are adjusted for better visibility.
    """

    # Load data from Excel file
    df_age_plot = pd.read_excel(input_file)
    df_age_plot.loc[(df_age_plot['Zircon'] == "TTG zircon") | (df_age_plot['Zircon'] == "I-type zircon"), "Zircon"] = "non-S-type zircon"


    # Define zircon types, colors, shapes, face colors, alpha values, and z-order
    zircon_types = [
        'Detrital zircon',
        "JH zircon",
        'S-type zircon',
        'non-S-type zircon',
        # 'Archean detrital zircon'
    ]

    edge_colors = [
        'none',
        'none',
        'steelblue',
        'red',
        'none'
    ]

    markers = [
        '.',
        '.',
        'o',
        'o',
        '.'
    ]

    face_colors = [
        'gray',
        'orange',
        'none',
        'none',
        'gray'
    ]

    alphas = [0.3, 0.4, 1, 1, 1]

    z_order = [10, 10, 10, 10, 1]

    # Initialize the figure and axes
    fig, ax = plt.subplots(figsize=(4, 3))

    # Scatter plot for each zircon type
    for t, c, f, m, a, z in zip(zircon_types, edge_colors, face_colors, markers, alphas, z_order):
        subset_data = df_age_plot[df_age_plot["Zircon"] == t]
        plt.scatter(
            subset_data["Age(Ma)"] / 1000,
            subset_data["P (μmol/g)"],
            s=15,
            marker=m,
            facecolors=f,
            edgecolors=c,
            linewidth=1.0,
            alpha=a,
            label=t,
            clip_on=False,
            zorder=z,
        )

    # Customize legend
    lgnd = plt.legend()
    for handle in lgnd.legendHandles:
        handle.set_sizes([10.0])

    # Set plot boundaries and configure x and y-axis ticks
    plt.xlim(0, 4.5)
    plt.ylim(0, 70)
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")

    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=10))

    ax.tick_params(axis="y", direction="in", labelsize=8)
    ax.tick_params(axis="x", direction="in", labelsize=8)

    # Add axis labels
    plt.xlabel('Age (Ga)', fontsize=8)
    plt.ylabel('P (μmol/g)', fontsize=8)

    # Save and display the figure
    plt.tight_layout()
    plt.savefig(f"{output_file}.jpg", dpi=300)
    plt.savefig(f"{output_file}.pdf")
    plt.show()


def plot_fig3A(input_data, method="tsvm", output_file=None):
    """
    This function compares the performance of a given method (default: tsvm) with a traditional method,
    by analyzing zircon data from an Excel file. It calculates and plots the classification accuracy
    for both methods in terms of P-criterion and the specified model across different P (μmol/g) bins.

    Parameters:
    input_file (str): Path to the Excel file containing zircon data.
    method (str): The name of the machine learning method being evaluated (default: 'tsvm').
    output_file (str, optional): Base filename to save the resulting plot as PNG and PDF files.

    Description:
    1. Reads the zircon data into a DataFrame.
    2. Assigns I-type and S-type labels based on the prediction set or training/test sets.
    3. Evaluates the correctness of the predictions using the provided method.
    4. Computes the number of I-type and S-type zircons per P bin.
    5. Calculates the accuracy of P-criterion and the specified model for each type and bin.
    6. Plots bar charts and scatter points representing the counts and accuracies respectively.

    Returns:
    A figure comparing the performances of the two methods across different P bins.
    """
    # Map predicted labels to model names for all sets
    input_data.loc[(input_data["Set"] == "Prediction set") & (input_data[method + ' Label'] == 0), method + " model"] = "non-S-type detrital zircon"
    input_data.loc[(input_data["Set"] == "Prediction set") & ([method + ' Label'] == 1), method + " model"] = "S-type detrital zircon"
    input_data.loc[(input_data["Set"] == "Training set") & (input_data[method + ' Label'] == 0), method + " model"] = "non-S-type zircon"
    input_data.loc[(input_data["Set"] == "Training set") & (input_data[method + ' Label'] == 1), method + " model"] = "S-type zircon"
    input_data.loc[(input_data["Set"] == "Testing set") & (input_data[method + ' Label'] == 0), method + " model"] = "non-S-type zircon"
    input_data.loc[(input_data["Set"] == "Testing set") & (input_data[method + ' Label'] == 1), method + " model"] = "S-type zircon"

    # Evaluate model performance
    #input_data[f'Performance of {method} model'] = np.where(input_data['Zircon'] == input_data[method + ' model'],
    #                                                         'correct', 'wrong')

    # Filter out non-training/testing set data
    data = input_data[input_data["Set"].isin(['Training set', "Testing set"])]
    data.loc[(data['Zircon'] == "TTG zircon") | (data['Zircon'] == "I-type zircon"), "Zircon"] = "non-S-type zircon"

    # Initialize statistics DataFrame
    stat = pd.DataFrame()

    # 1. Count I-type and S-type zircons in P bins
    bins = np.linspace(0, 70, 15)
    data["P bins"] = pd.cut(data["P (μmol/g)"], bins)
    zircon_num_bins = data.groupby(["Zircon", "P bins"])["P bins"].count()
    stat["non-S-type zircon"] = zircon_num_bins["non-S-type zircon"]
    stat["S-type zircon"] = zircon_num_bins["S-type zircon"]

    # 2. Calculate P-criterion accuracy
    S_type_data = data[data["Zircon"] == "S-type zircon"]
    count_bins1 = S_type_data.groupby(["Performance of P criteria", "P bins"])["P bins"].count()
    P_acc_bins_S = count_bins1 / count_bins1.groupby(level=[1]).transform(sum)
    stat["S type P criterion"] = P_acc_bins_S['correct']

    non_S_type_data = data[data["Zircon"] == "non-S-type zircon"]
    count_bins1 = non_S_type_data.groupby(["Performance of P criteria", "P bins"])["P bins"].count()
    P_acc_bins_I = count_bins1 / count_bins1.groupby(level=[1]).transform(sum)
    stat["non-S type P criterion"] = P_acc_bins_I['correct']

    # 3. Calculate model accuracy
    count_bins2_S = S_type_data.groupby([f"Performance of {method} model", "P bins"])["P bins"].count()
    SVM_acc_bins_S = count_bins2_S / count_bins2_S.groupby(level=[1]).transform(sum)
    stat[f"S type {method} model"] = SVM_acc_bins_S['correct']

    count_bins2_I = non_S_type_data.groupby([f"Performance of {method} model", "P bins"])["P bins"].count()
    SVM_acc_bins_I = count_bins2_I / count_bins2_I.groupby(level=[1]).transform(sum)
    stat[f"non-S type {method} model"] = SVM_acc_bins_I['correct']

    # Plotting
    fig1, ax1 = plt.subplots(figsize=(4, 3))

    # Bar chart for I-type and S-type zircon counts
    x_positions = np.linspace(2.5, 67.5, len(stat))
    y_values = stat["S-type zircon"]
    ax1.bar(x_positions, y_values, width=4)
    ax1.tick_params(axis="y", direction="in")
    ax1.tick_params(axis="x", direction="in")
    ax1.yaxis.set_ticks_position('right')
    ax1.spines['left'].set_color('none')
    ax1.spines['top'].set_color("none")
    ax1.axhline(0, color="grey")

    scale_ratio = 3
    y_values_scaled = stat["non-S-type zircon"] / scale_ratio
    ax1.bar(x_positions, -y_values_scaled, width=4, color='red')

    plt.yticks([-80 / scale_ratio, -40 / scale_ratio, 0, 10, 20, 30], ["80", "40", 0, 10, 20, 30])
    plt.autoscale(enable=True, axis='y')
    ax1.set_ylim([-80 / scale_ratio, 30])
    ax1.set_xlim([0, 70])
    ax1.xaxis.set_ticks_position('bottom')
    ax1.spines['bottom'].set_position(('data', -80 / scale_ratio - 2))
    ax1.annotate('S-type zircon',
                 xy=(260, 200), xycoords='figure pixels', color="steelblue")
    ax1.annotate('non-S-type zircon',
                 xy=(180, 100), xycoords='figure pixels', color="red")

    # Accuracy line plot for both types
    ax2 = ax1.twinx()
    ax2.tick_params(axis="y", direction="in")
    ax2.tick_params(axis="x", direction="in")
    ax2.spines['top'].set_color("none")
    ax2.spines['bottom'].set_color("none")

    x_positions -= 1.7
    ax2.scatter(x_positions, stat[f"S type {method} model"], marker="_", color="green")
    x_positions += 3.4
    ax2.scatter(x_positions, stat[f"S type {method} model"], marker="_", color="green")
    x_positions -= 1.7
    ax2.scatter(x_positions, stat[f"S type {method} model"], marker="o", edgecolor="green", facecolor='none',
                label=f"{method} model")
    ax2.scatter(x_positions, stat["S type P criterion"], marker="x", s=18, color="black", linewidth=0.8,
                label="P criterion")

    x_positions -= 1.7
    ax2.scatter(x_positions, -stat[f"non-S type {method} model"], marker="_", color="green")
    x_positions += 3.4
    ax2.scatter(x_positions, -stat[f"non-S type {method} model"], marker="_", color="green")
    x_positions -= 1.7
    ax2.scatter(x_positions, -stat[f"non-S type {method} model"], marker="o", edgecolor="green", facecolor='none')
    ax2.scatter(x_positions, -stat["non-S type P criterion"], marker="x", s=18, color="black", linewidth=0.8)

    ax2.set_ylim([-1.1, 1.1])
    ax2.set_xlim([0, 70])
    plt.yticks([-1.0, -0.9, 0, 0.5, 1.0], ["1.0", "0.9", 0, 0.5, 1.0])
    plt.legend(loc='best', frameon=False)

    if output_file is not None:
        fig1.savefig(f"{output_file}.png")
        fig1.savefig(f"{output_file}.pdf")

    plt.show()


def plot_fig3B(input_data, output_file):
    """
    This function plots the TSVM score against P (μmol/g) values for various types of zircon,
    including detrital, S-type, I-type, and their detrital subtypes. The result is a scatter plot
    with different colors and transparency levels to distinguish between each zircon type.

    Parameters:
    input_file (str): Path to the Excel file containing the tsvm scores and P values data.
    output_file (str): Base filename for saving the resulting plot as JPG and PDF files.

    Description:
    1. Reads the data from an Excel sheet into a Pandas DataFrame.
    2. Consolidates some zircon types under 'Detrital zircon'.
    3. Defines colors and facecolors for each zircon type in the scatter plot.
    4. Plots the TSVM value against P (μmol/g) for each zircon type.
    5. Configures axis limits, tickers, labels, and legend.
    6. Saves the plot to disk and displays it interactively.
    """

    # Combine certain zircon categories
    input_data.loc[input_data["Zircon"].isin(["Archean detrital zircon", "JH zircon", "GSB zircon"]), "Zircon"] = "Detrital zircon"
    input_data.loc[(input_data['Zircon'] == "TTG zircon") | (input_data['Zircon'] == "I-type zircon"), "Zircon"] = "non-S-type zircon"


    # Define zircon types and corresponding styles
    zircon_types = [
        'Detrital zircon',
        'S-type zircon',
        'non-S-type zircon',
        'non-S-type detrital zircon',
        'S-type detrital zircon',
    ]
    edge_colors = [
        'none',
        'steelblue',
        'red',
        'steelblue',
        'red',
    ]
    face_colors = [
        '#FFA500',  # Orange (replacing commented out color)
        'white',
        'white',
        'none',
        'none'
    ]
    alpha_values = [0.5, 1.0, 1.0]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(4, 3))

    # Plot scatter points for each zircon type
    for zircon_type, ec, fc, a in zip(zircon_types, edge_colors, face_colors, alpha_values):
        subset_data = input_data[input_data["Zircon"] == zircon_type]
        ax.scatter(
            subset_data["P (μmol/g)"],
            subset_data["tsvm value"],
            s=20,
            facecolors=fc,
            edgecolors=ec,
            linewidth=1,
            alpha=a,
            label=zircon_type
        )

    # Draw dashed horizontal line at y=0
    plt.hlines(0, 0, 70, linestyles="dashed", colors="k", lw=1)

    # Configure legend
    legend = plt.legend(fontsize=8)
    for handle in legend.legendHandles:
        handle.set_sizes([10])

    # Set axis limits and major locators
    ax.set_xlim(left=0, right=70)
    ax.set_ylim(bottom=-20, top=20)
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=10))
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=5))

    # Set axis tick parameters and labels
    ax.tick_params(axis="y", direction="in", labelsize=8)
    ax.tick_params(axis="x", direction="in", labelsize=8)
    plt.xlabel('P (μmol/g)', fontsize=8)
    plt.ylabel('TSVM value', fontsize=8)

    # Save the plot to disk and display it
    plt.tight_layout()
    plt.savefig(f"{output_file}.jpg")
    plt.savefig(f"{output_file}.pdf")
    plt.show()


def plot_fig4C(S_ratio_seq, output_file):
    """
    This function plots the ratio of S-type zircons in JH zircons using bootstrap resampling.
    It calculates and plots the mean 'S ratio' with confidence intervals over time intervals based on 'Age (Ma)'.

    Parameters:
    input_file (str): Path to the Excel file containing data on JH zircons.
    output_file (str): Base filename for saving the resulting plot as JPG and PDF files.

    Description:
    1. Reads the JH zircon data from an Excel sheet into a Pandas DataFrame.
    2. Computes the median age and mean 'S ratio' with their standard deviations over time sequences.
    3. Plots the mean 'S ratio' as a line plot and visualizes the confidence interval with filled area.
    4. Configures axis limits, tickers, aspect ratio, and removes unnecessary spines.
    5. Saves the plot to disk and displays it interactively.
    """

    # Initialize figure and axes
    fig, ax = plt.subplots(figsize=(9, 3))

    # Set colors and other parameters
    color = "orange"
    alpha_fill = 0.3
    error_alpha = 0.3
    s_ratio_type = "S ratio"

    # Calculate normalized values and confidence intervals
    x_values = S_ratio_seq["AGE_MEDIAN"] / 1000
    y_mean = S_ratio_seq[f"{s_ratio_type} mean"] * 100
    y_std = S_ratio_seq[f"{s_ratio_type} std"] * 100
    y_min = y_mean - y_std
    y_max = y_mean + y_std

    # Plot the mean S ratio line
    ax.plot(x_values, y_mean, color=color, label=f"{s_ratio_type}")

    # Plot confidence interval lines
    ax.plot(x_values, y_min, linestyle="--", alpha=error_alpha, color='grey')
    ax.plot(x_values, y_max, linestyle="--", alpha=error_alpha, color='grey')

    # Fill between confidence interval
    ax.fill_between(x_values, y_min, y_max, color=color, alpha=alpha_fill)

    # Configure axes limits and ticks
    ax.set_xlim(3.1, 4.5)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(MultipleLocator(base=0.2))
    ax.yaxis.set_major_locator(MultipleLocator(base=25))

    # Customize spines visibility
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Axis labels and ticks direction
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    ax.set_ylabel("S-type zircon (%)")
    ax.set_aspect('auto')


    # Save and display the plot
    plt.savefig(output_file + ".pdf")
    plt.savefig(output_file + ".jpg")
    plt.show()


def plot_fig4D(input_data, output_file):
    """
    This function reads an Excel file and plots a stacked bar graph for JH Zircon ages.

    Args:
    input_file (str): The path to the Excel file containing data for plotting.
    output_file (str): The base name of the output image files.

    Description:
    Fig. 4D: Creates a stacked bar graph showing the distribution of ages for different types of Jack Hills zircon.

    """

    # Define types of zircon
    ml_ziron_types = [
        'non-S-type detrital zircon',
        'non-S-type Jack Hills zircon',
        'S-type detrital zircon',
        'S-type Jack Hills zircon'
    ]

    # Define colors for each type
    colors_JH = ['#FF9587', '#FB696A', '#9CC1E0', '#2F92D7']

    # Set age bins
    bins = range(2500, 4550, 50)

    # Extract age lists for each zircon type
    x_data = [input_data[input_data['tsvm model'] == t]['Age(Ma)'].tolist() for t in ml_ziron_types]

    # Plotting settings
    fig, ax = plt.subplots(figsize=(6, 3.5))

    # Stacked histogram with custom labels and colors
    ax.hist(x_data, bins=bins, stacked=True, color=colors_JH, edgecolor="black", linewidth=0.1,
            label=ml_ziron_types)

    # Add legend
    plt.legend(loc='upper right')

    # Axis labels
    plt.xlabel('Age (Ma)', fontsize=13)
    plt.ylabel('Frequency', fontsize=13)

    # Tweak tick locations
    ax.yaxis.set_major_locator(MultipleLocator(base=50))
    ax.set_xlim(left=2500, right=4500)
    ax.set_xticks(range(2500, 4600, 200))
    ax.set_xticklabels([f"{round(x, 1)}" for x in np.arange(2.5, 4.7, 0.2)])
    ax.set_ylim(bottom=0, top=250)

    # Tick parameters
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")

    # Save figure in both JPG and PDF formats
    plt.savefig(output_file + '.jpg')
    plt.savefig(output_file + '.pdf')

    # Display the plot
    plt.show()


def plot_fig5(global_S_ratio_seq, Jh_S_ratio_seq, output_file):
    """
    This function loads zircon data from an Excel file and computes the ratio of S-type zircons
    globally and for Jack Hills (JH) specifically. It then plots these ratios over time
    with error bars and saves the figure in both PDF and JPG formats.

    Args:
    input_files (List[str]): A list containing two paths to Excel files.
        The first one contains the main dataset, while the second has prediction data.
    output_file (str): The base name of the output image files.

    Dependencies:
    - pandas
    - statsmodels (for lowess smoother)
    - scikit-learn (for model loading and predictions)

    Note: Ensure that 'elements' and 'elements_brev' variables are defined elsewhere.
    """
    # Combine sequences
    global Type
    Type = "S ratio"
    S_ratio_seq = global_S_ratio_seq.combine_first(Jh_S_ratio_seq)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 4))

    # Commented out specific plotting functions for brevity; assume they plot the data appropriately
    plot_whole_earth_error_bar(ax, global_S_ratio_seq, color="steelblue")
    plot_whole_earth_error_bar(ax, Jh_S_ratio_seq, color="orange")
    plot_whole_earth_curve(ax, S_ratio_seq)

    # Save the figure
    plt.savefig(output_file + ".pdf")
    plt.savefig(output_file + ".jpg")

    # Display the plot
    plt.show()


def plot_whole_earth_error_bar(ax, S_ratio_seq, color="steelblue"):
    """
    Plots the global S-type zircon ratio with error bars on the provided axis.

    Args:
    ax (matplotlib.axes.Axes): The axis to plot on.
    S_ratio_seq (pandas.DataFrame): DataFrame containing age and proportion data.
    color (str, optional): Color of the plotted elements. Default is "steelblue".

    This function plots the mean proportion of S-type zircons over time,
    using the median age and standard deviation from `S_ratio_seq`. It also performs
    a LOWESS smoothing and sets labels and axes limits.
    """

    # Set major and minor tick locators for both x and y axes
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
    ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.5))
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.2))

    # Define x-axis values and clean up NaNs
    x = S_ratio_seq["AGE_MEDIAN"] / 1000
    y = S_ratio_seq[f"{Type} mean"]
    error = S_ratio_seq[f"{Type} std"]
    valid_indices = ~np.isnan(y)
    x_clean = x[valid_indices]
    y_clean = y[valid_indices]
    error_clean = error[valid_indices]

    # Plot error bars
    plt.errorbar(x_clean, y_clean, yerr=error_clean, fmt='o', markersize=2, color=color, ecolor=color, capsize=2, lw=1)

    # Smoothed trend (LOWESS)
    smoothed = lowess(y_clean, x_clean, frac=0.08)
    fitted_x, fitted_y = smoothed[:, 0], smoothed[:, 1]

    # Format axis labels and ticks
    ax.set_xlabel("AGE(Ga)", fontsize=10, labelpad=10, weight='normal')
    ax.set_ylabel("S-type zircon proportion", fontsize=10, labelpad=10, weight='normal')
    ax.tick_params(labelsize=10, color='black')
    plt.xticks(fontsize=10, color='black')
    plt.yticks(fontsize=10, color='black')

    # Tighten layout
    plt.tight_layout()


def plot_whole_earth_curve(ax, S_ratio_seq):
    """
    Plots the global S-type zircon ratio curve with confidence intervals on the provided axis.

    Args:
    ax (matplotlib.axes.Axes): The axis to plot on.
    S_ratio_seq (pandas.DataFrame): DataFrame containing age and proportion data.

    This function calculates a LOWESS smooth of the data and creates confidence intervals,
    then plots the smoothed trend with shaded envelopes.
    """

    # Process data and perform LOWESS smoothing
    x = S_ratio_seq["AGE_MEDIAN"] / 1000
    y = S_ratio_seq[f"{Type} mean"]
    sem = S_ratio_seq[f"{Type} sem"]
    valid_indices = ~np.isnan(y)
    x_clean = x[valid_indices]
    y_clean = y[valid_indices]

    smoothed = lowess(y_clean, x_clean, frac=0.08)
    fitted_x, fitted_y = smoothed[:, 0], smoothed[:, 1]

    # Interpolate and calculate confidence intervals
    fitted_xnew = np.linspace(fitted_x.min(), fitted_x.max(), 300)
    func = interp1d(fitted_x, fitted_y, kind='quadratic')
    fitted_ynew = func(fitted_xnew)
    sem_new = np.std(y_clean-fitted_y) / np.sqrt(len(x))

    ci_68_lower = fitted_ynew - 1.0 * sem_new
    ci_68_upper = fitted_ynew + 1.0 * sem_new
    ci_95_lower = fitted_ynew - 1.96 * sem_new
    ci_95_upper = fitted_ynew + 1.96 * sem_new

    # Plot the smoothed curve and confidence intervals
    ax.plot(fitted_xnew, fitted_ynew, color='grey', label='Smoothed Trend')
    ax.fill_between(fitted_xnew, ci_68_lower, ci_68_upper, color='grey', alpha=0.2, label='68% CI')
    ax.fill_between(fitted_xnew, ci_95_lower, ci_95_upper, color='grey', alpha=0.1, label='95% CI')


def plot_JH_S_ratio(ax, S_ratio_seq, color="steelblue"):
    """
    Plots the ratio of S-type zircons in JH zircons with bootstrapped confidence bands.

    Args:
    ax (matplotlib.axes.Axes): The axis to plot on.
    S_ratio_seq (pandas.DataFrame): DataFrame containing age and proportion data.
    color (str, optional): Color of the plotted elements. Default is "steelblue".

    This function plots the mean proportion of S-type zircons from JH dataset,
    highlighting geological periods with grey shading and showing uncertainty as shaded regions.
    """

    # Define geologic periods
    periods = [
        (0.225, 0.35),
        (0.95, 1.25),
        (0.4, 0.65),
        (1.6, 1.95),
        (2.60, 2.75),
        (0.635, 0.72),
        (2.1, 2.4),
    ]

    for start, end in periods:
        ax.axvspan(start, end, ymin=0, ymax=1, facecolor="grey", alpha=0.05)

    # Plot data points with error bars
    x = S_ratio_seq["AGE_MEDIAN"] / 1000
    y = S_ratio_seq[f"{Type} mean"]
    std = S_ratio_seq[f"{Type} std"]

    ax.plot(x, y, color=color, lw=1, alpha=0.6)
    y_min = y - std
    y_max = y + std

    line1 = plt.plot(x, y_min, linestyle="--", color='grey', lw=1, alpha=0.6)
    line2 = plt.plot(x, y_max, linestyle="--", color='grey', lw=1)

    # Shade between error bars
    ax.fill_between(x, y_min, y_max, color=color, alpha=0.1, label=Type)

    # Set axis limits and labels
    ax.set_xlim(0, 4.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("AGE(Ga)", fontsize=10, labelpad=10, weight='normal')
    ax.set_ylabel("S-type zircon proportion", fontsize=10, labelpad=10, weight='normal')
    ax.tick_params(labelsize=10, color='black')
    plt.xticks(fontsize=10, color='black')
    plt.yticks(fontsize=10, color='black')

    plt.tight_layout()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.05, 1.0, 20), verbose=0):

    f, ax = plt.subplots(figsize=(4, 3))
    if ylim is not None:
        plt.ylim(*ylim)

    plt.grid(ls='--')
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, random_state =2, verbose=verbose)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_scores_mean, marker='o', color="r",
             label="Training accuracy",alpha=0.6, markersize='3')

    plt.plot(train_sizes, test_scores_mean, marker='o', color="#F97306",
             label="Cross-validation accuracy", alpha=0.6,  markersize='3')


    plt.xlabel("The number of samples in training set", fontsize=8)
    plt.ylabel("Accuracy", fontsize=8)
    plt.xticks( fontsize=8)
    plt.yticks(fontsize=8)
    # ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=20))
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.2))
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    plt.xlim(20, 160)
    plt.ylim(0.4, 1.1)
    plt.tight_layout()
    #plt.plot([0, 160], [0.9, 0.9], color="k", linestyle='--',lw=0.5)
    plt.legend(loc="lower right", fontsize=8)
    savePath = fig_path + "Learning_curve/"
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    plt.draw()
    #plt.gca().invert_yaxis()
    plt.savefig(savePath + title + ".png", dpi=300)
    plt.savefig(savePath + title + ".pdf")
    plt.show()
    return plt


if __name__ == "__main__":
    # Create the directory for saving plots if it doesn't exist already
    os.makedirs(fig_path, exist_ok=True)

    # Constants for file paths
    base_input_file = output_path + file_name + "_with_prediction.xlsx"
    whole_earth_data = data_path + "Global_detrital_zircon_data.xlsx"

    zircons_data = pd.read_excel(base_input_file)


    # Fig. 1A: Plot (REE+Y)3 vs Y
    plot_fig1A(input_file=base_input_file,
               output_file=fig_path + "Fig1A_REE_P_with_detrital_zircon")

    # Fig. 1B: Plot Age vs P
    plot_fig1B(input_file=data_path + "database 2 with_whole_earth_zircon.xlsx",
               output_file=fig_path + "Fig1B_Age_P20230924")


    # Fig. 3A: Compare TSVM performance with traditional method
    tsvm_method = "tsvm"

    plot_fig3A(input_data=zircons_data.copy(),
               method=tsvm_method,
               output_file=fig_path + f"Fig3A_Accuracy_of_{tsvm_method}_vs_traditional_method")

    # Fig. 3B: Plot TSVM score for detrital zircons
    
    plot_fig3B(input_data=zircons_data.copy(),
               output_file=fig_path + "Fig3B_TSVM_Score_with_type0706")

    # Fig. 4C: Plot S-type zircon ratio for JH zircon sample
    Jh_S_ratio_seq = pd.read_csv(output_path + "Bootstrap_means_JH_zircon_" + tsvm_method + ".csv")
    plot_fig4C(S_ratio_seq=Jh_S_ratio_seq,
               output_file=fig_path + "Fig4C_Ratio_of_S-type_Zircons")


    # Fig. 4C: Plot S-type zircon ratio for GSSB zircon sample
    GSSB_S_ratio_seq = pd.read_csv(output_path + "Bootstrap_means_GSSB_zircon_" + tsvm_method + ".csv")
    plot_fig4C(S_ratio_seq=GSSB_S_ratio_seq,
               output_file=fig_path + "Fig4C_Ratio_of_GSSB_S-type_Zircons")

    # Fig. 4D: Bar graph for detrital Zircon data
    detrital_data = zircons_data[(zircons_data["Set"] == "Prediction set")].copy()
    detrital_data.loc[(zircons_data["Zircon"] == "JH zircon") & (zircons_data["tsvm model"] == "S-type detrital zircon"), "tsvm model"] = "S-type Jack Hills zircon"
    detrital_data.loc[(zircons_data["Zircon"] == "JH zircon") & (zircons_data["tsvm model"] == "non-S-type detrital zircon"), "tsvm model"] = "non-S-type Jack Hills zircon"
    detrital_data.reset_index(inplace=True, drop=True)
    plot_fig4D(input_data=detrital_data,
               output_file=fig_path + "Fig4D_sed_stacked_hist")

    # Fig. 5: Global S-type zircon ratio using TSVM method
    tsvm_method = "tsvm"
    global_S_ratio_seq = pd.read_csv(output_path + "Bootstrap_means_global_detrital_zircon_data_" + tsvm_method + ".csv")
    plot_fig5(global_S_ratio_seq,
              Jh_S_ratio_seq,
              output_file=fig_path + f"Fig5D_{tsvm_method}_Global_S-Type_Zircon_Ratio")


