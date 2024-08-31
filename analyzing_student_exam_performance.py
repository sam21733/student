# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = '/workspaces/super-duper-garbanzo/StudentsPerformance.csv'
df = pd.read_csv(file_path)

"""## Initial Exploration"""

# Display the first few rows of the dataset
df.head()

"""## Data Cleaning and Preparation"""

# Check for missing values
df.isnull().sum()

# Check data types
df.dtypes

"""## Descriptive Statistics"""

# Summary statistics
df.describe()

"""## Visualizations"""

# Distribution of scores
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(df['math score'], kde=True)
plt.title('Math Score Distribution')
plt.subplot(1, 3, 2)
sns.histplot(df['reading score'], kde=True)
plt.title('Reading Score Distribution')
plt.subplot(1, 3, 3)
sns.histplot(df['writing score'], kde=True)
plt.title('Writing Score Distribution')
plt.show()

# Box plots for scores by gender
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.boxplot(x='gender', y='math score', data=df)
plt.title('Math Score by Gender')
plt.subplot(1, 3, 2)
sns.boxplot(x='gender', y='reading score', data=df)
plt.title('Reading Score by Gender')
plt.subplot(1, 3, 3)
sns.boxplot(x='gender', y='writing score', data=df)
plt.title('Writing Score by Gender')
plt.show()

"""## Correlation Analysis"""

# Correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

"""## Predictive Modeling"""

# Prepare data for modeling
X = df[['math score', 'reading score']]
y = df['writing score']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse, r2

"""## Conclusion and Future Work

In this notebook, we explored the Students Performance in Exams dataset, visualized the distributions and relationships between scores, and built a simple predictive model for writing scores based on math and reading scores. The model's performance can be further improved by exploring more sophisticated algorithms or feature engineering.

What other analyses do you think would be useful? Feel free to share your thoughts and if you found this notebook helpful, please upvote it.

## Credits
This notebook was created with the help of [Devra AI data science assistant](https://devra.ai/ref/kaggle)
"""