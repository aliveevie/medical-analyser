import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')
df = df[(df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))]


# Add 'overweight' column
df['BMI'] = df['weight'] / (df['height'] / 100) ** 2
df['overweight'] = (df['BMI'] > 25).astype(int)
# Normalize cholesterol and gluc columns
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.


# Draw Categorical Plot
def draw_cat_plot():
    corr_matrix = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, mask=mask, cmap='coolwarm', fmt=".2f", square=True)

    # Set plot title
    plt.title('Correlation Matrix')

    # Display the plot
    fig = plt.show()


    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
 
    # Clean the data
    plt.figure(figsize=(10, 8))
    

    # Calculate the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    df_heat = sns.heatmap(corr, annot=True, mask=mask, cmap='coolwarm', fmt=".2f", square=True)



    # Set up the matplotlib figure
    ax = plt.title('Correlation Matrix')

    # Draw the heatmap with 'sns.heatmap()'
    fig = plt.show()


    # Do not modify the next two lines
    #fig.savefig('heatmap.png')
    return fig
