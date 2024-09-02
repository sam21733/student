# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load the dataset
df = pd.read_csv('StudentsPerformance.csv')  # Update the path if needed

# Initial Exploration: Display the first few rows of the dataset
print(df.head())

# Data Cleaning and Preparation: Check for missing values
print(df.isnull().sum())

# Check data types
print(df.dtypes)

# Descriptive Statistics: Summary statistics
print(df.describe())

# Visualizations
plt.figure(figsize=(15, 5))

# Distribution of Math Score
plt.subplot(1, 3, 1)
sns.histplot(df['math score'], kde=True)
plt.title('Math Score Distribution')

# Distribution of Reading Score
plt.subplot(1, 3, 2)
sns.histplot(df['reading score'], kde=True)
plt.title('Reading Score Distribution')

# Distribution of Writing Score
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

# Correlation Analysis: Correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# Predictive Modeling: Prepare data for modeling
X = df[['math score', 'reading score']]
y = df['writing score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Save the trained model to a pickle file
with open('StudentsPerformance.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as StudentsPerformance.pkl")
