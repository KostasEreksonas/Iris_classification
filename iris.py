#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy.stats import norm # Used to plot normal distribution
import matplotlib.pyplot as plt

def data_preparation():
    # Load columns
    columns = pd.read_csv("data/iris names.txt", names=['names'])
    # Load data
    data = pd.read_csv("data/iris2.data", sep="\t", header=None, names=columns['names'])
    # Separate dataframe by class number (for plotting graphs)
    class_1 = data[data['class'] == 1]
    class_2 = data[data['class'] == 2]
    class_3 = data[data['class'] == 3]
    # Get labels of the dataset
    true_class = data['class']
    data = data.drop('class', axis=1)
    # Get parameter stats of every class
    c1s = class_stats(class_1)
    c2s = class_stats(class_2)
    c3s = class_stats(class_3)
    return columns, true_class, data, class_1, class_2, class_3, c1s, c2s, c3s

def class_stats(df):
    return df['sepal length in cm'].min(), \
           df['sepal length in cm'].max(), \
           df['sepal length in cm'].mean(), \
           df['sepal length in cm'].std(), \
           df['sepal width in cm'].min(), \
           df['sepal width in cm'].max(), \
           df['sepal width in cm'].mean(), \
           df['sepal width in cm'].std(), \
           df['petal length in cm'].min(), \
           df['petal length in cm'].max(), \
           df['petal length in cm'].mean(), \
           df['petal length in cm'].std(), \
           df['petal width in cm'].min(), \
           df['petal width in cm'].max(), \
           df['petal width in cm'].mean(), \
           df['petal width in cm'].std()

def print_stats(class_1, class_2, class_3, count = 0, name = ''):
    for x in range(3):
        if count == 0:
            arg = class_1
            name = 'class 1'
        elif count == 1:
            arg = class_2
            name = 'class 2'
        elif count == 2:
            arg = class_3
            name = 'class 3'
        print(f" --------------------------")
        print(f"|  Statistics for {name}   |")
        print(f" --------------------------")
        print(f"| Min sepal length: {arg[0]:.4f}   |")
        print(f"| Max sepal length: {arg[1]:.4f}   |")
        print(f"| Sepal length mean: {arg[2]:.4f}  |")
        print(f"| Sepal length std: {arg[3]:.4f}   |")
        print(f"| Min sepal width: {arg[4]:.4f}    |")
        print(f"| Max sepal width: {arg[5]:.4f}    |")
        print(f"| Sepal width mean: {arg[6]:.4f}   |")
        print(f"| Sepal width std: {arg[7]:.4f}    |")
        print(f"| Min petal length: {arg[8]:.4f}   |")
        print(f"| Max petal length: {arg[9]:.4f}   |")
        print(f"| Petal length mean: {arg[10]:.4f}  |")
        print(f"| Petal length std: {arg[11]:.4f}   |")
        print(f"| Min petal width: {arg[12]:.4f}    |")
        print(f"| Max petal width: {arg[13]:.4f}    |")
        print(f"| Petal width mean: {arg[14]:.4f}   |")
        print(f"| Petal width std: {arg[15]:.4f}    |")
        print(f" ----------------------------")
        count += 1

def scatter_plot(df_1, df_2, df_3, rows, cols):
    # df_1 - parameters for class 1
    # df_2 - parameters for class 2
    # df_3 - parameters for class 3
    # rows - rows of the subplot grid
    # cols - columns of the subplot grid
    fig, ax = plt.subplots(nrows=rows,
                           ncols=cols,
                           figsize=(10,10))
    columns = df_1.columns.to_list()
    c = 0
    for i in range(rows):
        for j in range(cols):
            ax[i,j].scatter(x=df_1[columns[c]].index, y=df_1[columns[c]], facecolors='none', edgecolors='red')
            ax[i,j].scatter(x=df_2[columns[c]].index, y=df_2[columns[c]], facecolors='none', edgecolors='green')
            ax[i,j].scatter(x=df_3[columns[c]].index, y=df_3[columns[c]], facecolors='none', edgecolors='blue')
            ax[i,j].grid(which='Major', axis='both')
            ax[i,j].set(title=f"{' '.join(columns[c].split()[:2]).title()}",
                        xlabel="Samples",
                        axisbelow=True)
            ax[i,j].set(ylabel="Length, cm") if j/2 == 0 else ax[i,j].set(ylabel="Width, cm")
            ax[i,j].legend(["Class 1", "Class 2", "Class 3"], title="Class distribution")
            c += 1
    fig.savefig("images/parameter_distribution.png")
    print(f"\n --------")
    print(f"| Images |")
    print(f" --------")
    print(f"----------------------------------------------------")
    print(f"[+] Parameter distribution graph saved to images/parameter_distribution.png")

def normal_distribution(c1s, c2s, c3s, rows, cols, names):
    """
        Create a normal distribution subplot for every parameter.
        Parameters are separated into 3 groups according to iris class.
    """
    # c1s - parameter statistics for class 1
    # c2s - parameter statistics for class 2
    # c3s - parameter statistics for class 3
    # c1s[0] - min sepal length
    # c1s[1] - max sepal length
    # c1s[2] - sepal length mean
    # c1s[3] - sepal length standard deviation
    # c1s[4] - min sepal width
    # c1s[5] - max sepal width
    # c1s[6] - sepal width mean
    # c1s[7] - sepal width standard deviation
    # c1s[8] - min petal length
    # c1s[9] - max petal length
    # c1s[10] - petal length mean
    # c1s[11] - petal length standard deviation
    # c1s[12] - min petal width
    # c1s[13] - max petal width
    # c1s[14] - petal width mean
    # c1s[15] - petal width standard deviation
    # c1s[0] - min sepal length
    # c1s[1] - max sepal length
    # c1s[2] - sepal length mean
    # c1s[3] - sepal length standard deviation
    # c1s[4] - min sepal width
    # c1s[5] - max sepal width
    # c1s[6] - sepal width mean
    # c1s[7] - sepal width standard deviation
    # c1s[8] - min petal length
    # c1s[9] - max petal length
    # c1s[10] - petal length mean
    # c1s[11] - petal length standard deviation
    # c1s[12] - min petal width
    # c1s[13] - max petal width
    # c1s[14] - petal width mean
    # c1s[15] - petal width standard deviation
    # rows - rows of the subplot grid
    # cols - columns of the subplot grid
    # names - column names
    # Group parameter stats to their respective list
    param1 = [c1s[0], c1s[1], c2s[0], c2s[1], c3s[0], c3s[1], c1s[2], c2s[2], c3s[2], c1s[3], c2s[3], c3s[3]]
    param2 = [c1s[4], c1s[5], c2s[4], c2s[5], c3s[4], c3s[5], c1s[6], c2s[6], c3s[6], c1s[7], c2s[7], c3s[7]]
    param3 = [c1s[8], c1s[9], c2s[8], c2s[9], c3s[8], c3s[9], c1s[10], c2s[10], c3s[10], c1s[11], c2s[11], c3s[11]]
    param4 = [c1s[12], c1s[13], c2s[12], c2s[13], c3s[12], c3s[13], c1s[14], c2s[14], c3s[14], c1s[15], c2s[15], c3s[15]]
    c = 0
    fig, ax = plt.subplots(nrows=rows,
                           ncols=cols,
                           figsize=(10,10))
    for i in range(rows):
        for j in range(cols):
            if c == 0:
                arg = param1
            elif c == 1:
                arg = param2
            elif c == 2:
                arg = param3
            elif c == 3:
                arg = param4
            # x axis - range between min and max values, step - 0.01
            x_1 = np.arange(arg[0], arg[1], 0.01)
            x_2 = np.arange(arg[4], arg[5], 0.01)
            x_3 = np.arange(arg[8], arg[9], 0.01)
            # Normal distribution has 3 parameters - x axis range, parameter mean, parameter standard deviation
            ax[i,j].plot(x_1, norm.pdf(x_1, arg[2], arg[3]), c='red')
            ax[i,j].plot(x_2, norm.pdf(x_2, arg[6], arg[7]), c='green')
            ax[i,j].plot(x_3, norm.pdf(x_3, arg[10], arg[11]), c='blue')
            ax[i,j].set(title=f"{' '.join(names[c].split()[:2]).title()}",
                        ylabel="Probability Density",
                        axisbelow=True)
            ax[i,j].set(xlabel="Length, cm") if j/2 == 0 else ax[i,j].set(xlabel="Width, cm")
            ax[i,j].grid(which='Major',
                         axis='both')
            ax[i,j].legend(["Class 1", "Class 2", "Class 3"],
                           title="Class Distribution",
                           loc="upper right")
            c += 1
    fig.savefig("images/normal_distribution.png")
    print(f"[+] Normal distribution graph saved to images/normal_distribution.png")

def classifier_1(df, true_class, c1s, c2s, c3s):
    """Augalų klasifikavimas naudojant if sąlygas ir min max vertes"""
    predictions = []
    for i in df.index:
        if df['sepal length in cm'][i] >= c1s[0] and df['sepal length in cm'][i] <= c1s[1] and \
            df['sepal width in cm'][i] >= c1s[4] and df['sepal width in cm'][i] <= c1s[5] and \
            df['petal length in cm'][i] >= c1s[8] and df['petal length in cm'][i] <= c1s[9] and \
            df['petal width in cm'][i] >= c1s[12] and df['petal width in cm'][i] <= c1s[13]:
            predictions.append(1)
        elif df['sepal length in cm'][i] >= c2s[0] and df['sepal length in cm'][i] <= c2s[1] and \
            df['sepal width in cm'][i] >= c2s[4] and df['sepal width in cm'][i] <= c2s[5] and \
            df['petal length in cm'][i] >= c2s[8] and df['petal length in cm'][i] <= c2s[9] and \
            df['petal width in cm'][i] >= c2s[12] and df['petal width in cm'][i] <= c2s[13]:
            predictions.append(2)
        elif df['sepal length in cm'][i] >= c3s[0] and df['sepal length in cm'][i] <= c3s[1] and \
            df['sepal width in cm'][i] >= c3s[4] and df['sepal width in cm'][i] <= c3s[5] and \
            df['petal length in cm'][i] >= c3s[8] and df['petal length in cm'][i] <= c3s[9] and \
            df['petal width in cm'][i] >= c3s[12] and df['petal width in cm'][i] <= c3s[13]:
            predictions.append(3)
    new_df = pd.DataFrame(predictions, columns=['predicted_class'])
    new_df.insert(1, "true_class", true_class, True)
    return new_df

def classifier_2(df, true_class, c1s, c2s, c3s):
    """Iris classification using if-elif-else statements with mean and standard deviation"""
    predictions = []
    for i in df.index:
        if df['sepal length in cm'][i] >= c1s[2]-3*c1s[3] and df['sepal length in cm'][i] <= c1s[2]+3*c1s[3] and \
            df['sepal width in cm'][i] >= c1s[6]-3*c1s[7] and df['sepal width in cm'][i] <= c1s[6]+3*c1s[7] and \
            df['petal length in cm'][i] >= c1s[10]-3*c1s[11] and df['petal length in cm'][i] <= c1s[10]+3*c1s[11] and \
            df['petal width in cm'][i] >= c1s[14]-3*c1s[15] and df['petal width in cm'][i] <= c1s[14]+3*c1s[15]:
            predictions.append(1)
        elif df['sepal length in cm'][i] >= c2s[2]-3*c2s[3] and df['sepal length in cm'][i] <= c2s[2]+3*c2s[3] and \
            df['sepal width in cm'][i] >= c2s[6]-3*c2s[7] and df['sepal width in cm'][i] <= c2s[6]+3*c2s[7] and \
            df['petal length in cm'][i] >= c2s[10]-3*c2s[11] and df['petal length in cm'][i] <= c2s[10]+3*c2s[11] and \
            df['petal width in cm'][i] >= c2s[14]-3*c2s[15] and df['petal width in cm'][i] <= c2s[14]+3*c2s[15]:
            predictions.append(2)
        elif df['sepal length in cm'][i] >= c3s[2]-3*c3s[3] and df['sepal length in cm'][i] <= c3s[2]+3*c3s[3] and \
            df['sepal width in cm'][i] >= c3s[6]-3*c3s[7] and df['sepal width in cm'][i] <= c3s[6]+3*c3s[7] and \
            df['petal length in cm'][i] >= c3s[10]-3*c3s[11] and df['petal length in cm'][i] <= c3s[10]+3*c3s[11] and \
            df['petal width in cm'][i] >= c3s[14]-3*c3s[15] and df['petal width in cm'][i] <= c3s[14]+3*c3s[15]:
            predictions.append(3)
    new_df = pd.DataFrame(predictions, columns=['predicted_class'])
    new_df.insert(1, "true_class", true_class, True)
    return new_df

def count_errors(df):
    TP,TN,FP,FN = [0 for x in range(4)]
    for i in df.index:
        if df['predicted_class'][i] == df['true_class'][i]:
            TP += 1
            TN += 2
        else:
            FP += 1
            FN += 1
    P = len(df)
    N = 2*len(df)
    return [TP,TN,FP,FN,P,N]

def print_error(df, name):
    TP,TN,FP,FN,P,N = count_errors(df)
    print(f"\n ------------------------")
    print(f"| {name} |")
    print(f" ------------------------")
    print(f"[+] TP: {TP}")
    print(f"[+] TN: {TN}")
    print(f"[+] FP: {FP}")
    print(f"[+] FN: {FN}")
    print(f"[+] P: {P}")
    print(f"[+] N: {N}")

def calculate_metrics(df):
    TP,TN,FP,FN,P,N = count_errors(df)
    accuracy = (TP + TN) / (P + N)
    sensitivity = TP / P
    precision = TP / (TP + FP)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return [accuracy,sensitivity,precision,mcc]

def print_metrics(df, name):
    accuracy,sensitivity,precision,mcc = calculate_metrics(df)
    print(f"\n --------------------------")
    print(f"| {name} |")
    print(f" --------------------------")
    print(f"[+] Accuracy: {accuracy*100:.2f}%")
    print(f"[+] Sensitivity: {sensitivity*100:.2f}%")
    print(f"[+] Precision: {precision*100:.2f}%")
    print(f"[+] Matthew coefficient: {mcc*100:.2f}%")

def main():
    columns, true_class, data, class_1, class_2, class_3, c1s, c2s, c3s = data_preparation()
    print_stats(c1s, c2s, c3s)
    scatter_plot(class_1, class_2, class_3, 2, 2)
    normal_distribution(c1s, c2s, c3s, 2, 2, columns['names'].values.tolist())
    method_1 = classifier_1(data, true_class, c1s, c2s, c3s)
    print_error(method_1, "Stats for classifier 1")
    print_metrics(method_1, "Metrics for classifier 1")
    method_2 = classifier_2(data, true_class, c1s, c2s, c3s)
    print_error(method_2, "Stats for classifier 2")
    print_metrics(method_2, "Metrics for classifier 2")

if __name__ == "__main__":
    main()
