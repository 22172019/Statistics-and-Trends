"""
Statistics and Trends Assignment
---------------------------------
This script performs exploratory data analysis, visualization, and
statistical analysis on the Automobile dataset. It follows the
given template structure strictly and ensures PEP-8 compliance.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def preprocessing(df):
    """
    Preprocess the dataset:
    - Clean missing values
    - Handle outliers
    - Convert numeric columns
    - Display summary statistics and correlations
    """
    df = df.copy()
    df.replace('?', np.nan, inplace=True)
    df.dropna(subset=['price', 'horsepower', 'fuel-type'], inplace=True)

    numeric_cols = [
        'price', 'horsepower', 'city-mpg',
        'highway-mpg', 'engine-size', 'curb-weight'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=[c for c in numeric_cols if c in df.columns], inplace=True)

    # Remove outliers
    for col in ['price', 'horsepower']:
        if col in df.columns:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]

    print("===== DATA PREVIEW =====")
    print(df.head(), "\n")
    print("===== SUMMARY STATISTICS =====")
    print(df.describe(), "\n")
    print("===== CORRELATION MATRIX =====")
    print(df.corr(), "\n")

    return df


def plot_relational_plot(df):
    """
    Generate relational plots (scatter, line, and step).
    """
    # Scatter: horsepower vs price
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df, x='horsepower', y='price',
        hue='fuel-type', style='fuel-type',
        s=120, palette='Set2', edgecolor='black', alpha=0.85
    )
    plt.xlabel("Horsepower")
    plt.ylabel("Price")
    plt.title("Horsepower vs Price by Fuel Type")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("relational_plot.png")
    plt.close()


def plot_categorical_plot(df):
    """
    Generate categorical plots (bar, horizontal bar, histogram, pie).
    """
    # Bar plot: fuel-type vs average price
    plt.figure(figsize=(9, 6))
    if 'fuel-type' in df.columns:
        avg_fuel = df.groupby('fuel-type')['price'].mean().sort_values()
        sns.barplot(x=avg_fuel.index, y=avg_fuel.values,
                    palette='Set2', edgecolor='black')
        plt.ylabel("Average Price")
        plt.xlabel("Fuel Type")
        plt.title("Fuel Type vs Average Price")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("categorical_plot.png")
    plt.close()


def plot_statistical_plot(df):
    """
    Generate statistical plots (box, violin, pairplot).
    """
    # Boxplot: price by fuel-type
    plt.figure(figsize=(10, 7))
    if 'fuel-type' in df.columns:
        sns.boxplot(
            x='fuel-type', y='price', data=df, palette='Set2',
            linewidth=1.5, showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "white",
                       "markeredgecolor": "black", "markersize": 10}
        )
        sns.stripplot(x='fuel-type', y='price', data=df,
                      color='black', size=4, jitter=True, alpha=0.6)
        plt.title("Boxplot of Price by Fuel Type")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("statistical_plot.png")
    plt.close()


def statistical_analysis(df, col: str):
    """
    Calculate key statistical moments for a given numeric column.
    Returns mean, standard deviation, skewness, and excess kurtosis.
    """
    mean = df[col].mean()
    stddev = df[col].std()
    skew_val = df[col].skew()
    excess_kurtosis = df[col].kurtosis()  # pandas returns excess kurtosis
    return mean, stddev, skew_val, excess_kurtosis


def writing(moments, col):
    """
    Print interpretation of statistical analysis results.
    """
    mean, stddev, skew_val, kurt = moments
    print(f"For the attribute '{col}':")
    print(f"Mean = {mean:.2f}, Standard Deviation = {stddev:.2f}, "
          f"Skewness = {skew_val:.2f}, Excess Kurtosis = {kurt:.2f}")

    if skew_val > 0:
        skew_desc = "right-skewed"
    elif skew_val < 0:
        skew_desc = "left-skewed"
    else:
        skew_desc = "not skewed"

    if kurt > 0:
        kurt_desc = "leptokurtic (peaked)"
    elif kurt < 0:
        kurt_desc = "platykurtic (flat)"
    else:
        kurt_desc = "mesokurtic (normal-like)"

    print(f"The data is {skew_desc} and {kurt_desc}.\n")


def main():
    """
    Main function that loads data, preprocesses it,
    generates plots, performs statistical analysis,
    and prints results.
    """
    df = pd.read_csv("Automobile_data.csv")
    df = preprocessing(df)

    col = "price"

    # Generate plots
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    # Statistical analysis
    moments = statistical_analysis(df, col)
    writing(moments, col)


if __name__ == "__main__":
    main()
