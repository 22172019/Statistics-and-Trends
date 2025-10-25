"""
This is the template file for the statistics and trends assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

def plot_relational_plot(df):

    # Scatter Plot: Horsepower vs Price
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df, x='horsepower', y='price', hue='fuel-type', style='fuel-type',
        s=120, palette='Set2', edgecolor='black', alpha=0.85
    )
    top3 = df.nlargest(3, 'price')
    for _, row in top3.iterrows():
        plt.text(
            row['horsepower'] + 2, row['price'] + 500,
            f"${int(row['price']):,}", color='red', fontweight='bold'
        )
    plt.title("Horsepower vs Price by Fuel Type", fontsize=14)
    plt.xlabel("Horsepower")
    plt.ylabel("Price")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('relational_plot.png')
    plt.show()

    # Line Plot: City MPG vs Highway MPG
    plt.figure(figsize=(10, 6))
    sorted_data = df.sort_values('city-mpg')
    plt.plot(sorted_data['city-mpg'], sorted_data['highway-mpg'],
             marker='o', color='green')
    plt.title("City MPG vs Highway MPG")
    plt.xlabel("City MPG")
    plt.ylabel("Highway MPG")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Step Plot: Engine Size vs Price
    plt.figure(figsize=(10, 6))
    sorted_data = df.sort_values('engine-size')
    plt.step(sorted_data['engine-size'], sorted_data['price'],
             where='mid', color='purple')
    plt.title("Engine Size vs Price")
    plt.xlabel("Engine Size")
    plt.ylabel("Price")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    return
    return


def plot_categorical_plot(df):
    # Bar Plot: Fuel Type vs Average Price
    plt.figure(figsize=(9, 6))
    avg_fuel = df.groupby('fuel-type')['price'].mean().sort_values()
    sns.barplot(x=avg_fuel.index, y=avg_fuel.values,
                palette='Set2', edgecolor='black')
    for i, v in enumerate(avg_fuel.values):
        plt.text(i, v + 500, f"${int(v):,}", ha='center', fontweight='bold')
    plt.title("Fuel Type vs Average Price")
    plt.ylabel("Average Price")
    plt.xlabel("Fuel Type")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('categorical_plot.png')
    plt.show()

    # Horizontal Bar Plot: Body Style vs Average Price
    plt.figure(figsize=(9, 6))
    avg_body = df.groupby('body-style')['price'].mean().sort_values()
    plt.barh(avg_body.index, avg_body.values,
             color='skyblue', edgecolor='black')
    for i, v in enumerate(avg_body.values):
        plt.text(v + 500, i, f"${int(v):,}", va='center', fontweight='bold')
    plt.title("Body Style vs Average Price")
    plt.xlabel("Average Price")
    plt.ylabel("Body Style")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Histogram: Price Distribution
    plt.figure(figsize=(9, 6))
    plt.hist(df['price'], bins=15, color='orange',
             edgecolor='black', alpha=0.7)
    plt.title("Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Pie Chart: Drive Wheels Distribution
    drive_counts = df['drive-wheels'].value_counts()
    plt.figure(figsize=(7, 7))
    plt.pie(
        drive_counts.values, labels=drive_counts.index,
        autopct='%1.1f%%', startangle=140,
        colors=sns.color_palette('Set2')
    )
    plt.title("Drive Wheels Proportion")
    plt.tight_layout()
    plt.show()
    return


def plot_statistical_plot(df):
    # Boxplot: Price by Fuel Type
    plt.figure(figsize=(10, 7))
    sns.boxplot(
        x='fuel-type', y='price', data=df, palette='Set2',
        linewidth=1.5, showmeans=True,
        meanprops={
            "marker": "o", "markerfacecolor": "white",
            "markeredgecolor": "black", "markersize": 10
        }
    )
    sns.stripplot(x='fuel-type', y='price', data=df,
                  color='black', size=4, jitter=True, alpha=0.6)
    plt.title("Price by Fuel Type (Boxplot)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('statistical_plot.png')
    plt.show()

    # Violin Plot: Price by Body Style
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='body-style', y='price', data=df,
                   palette='Set2', inner='quartile')
    plt.title("Price by Body Style (Violin Plot)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Pairplot
    sns.pairplot(df[['price', 'horsepower', 'engine-size', 'city-mpg']],
                 diag_kind='kde', plot_kws={'alpha': 0.6})
    plt.suptitle("Pairplot of Selected Numeric Features",
                 fontsize=16, weight='bold', y=1.02)
    plt.show()
    return


def statistical_analysis(df, col: str):
    mean = df[col].mean()
    stddev = df[col].std()
    skewness = skew(df[col])
    excess_kurtosis = kurtosis(df[col])
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
 df.replace('?', pd.NA, inplace=True)
    df.dropna(subset=['price', 'horsepower', 'fuel-type'], inplace=True)

    numeric_cols = [
        'price', 'horsepower', 'city-mpg',
        'highway-mpg', 'engine-size', 'curb-weight'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numeric_cols, inplace=True)

    for col in ['price', 'horsepower']:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]

    print("\nData Preview:\n", df.head())
    print("\nSummary Statistics:\n", df.describe())
    print("\nCorrelation Matrix:\n", df.corr(numeric_only=True))
    return df


def writing(moments, col):
print(f'\nFor the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, '
          f'Excess Kurtosis = {moments[3]:.2f}.')

    if moments[2] > 0:
        skew_type = "right-skewed"
    elif moments[2] < 0:
        skew_type = "left-skewed"
    else:
        skew_type = "symmetrical"

    if moments[3] < 0:
        kurt_type = "platykurtic (flatter tails)"
    elif -2 <= moments[3] <= 2:
        kurt_type = "mesokurtic (normal-like)"
    else:
        kurt_type = "leptokurtic (heavy tails)"

    print(f"The data is {skew_type} and {kurt_type}.")
    return


def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = 'price'

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)
    return


if __name__ == '__main__':
    main()
