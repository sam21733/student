import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv('student_scores.csv')

# Encode categorical variables
label_encoders = {}
for column in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split data into features and target variables
X = data[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']]
y_math = data['math score']
y_reading = data['reading score']
y_writing = data['writing score']

# Split data into training and testing sets
X_train, X_test, y_math_train, y_math_test = train_test_split(X, y_math, test_size=0.2, random_state=42)
X_train, X_test, y_reading_train, y_reading_test = train_test_split(X, y_reading, test_size=0.2, random_state=42)
X_train, X_test, y_writing_train, y_writing_test = train_test_split(X, y_writing, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Train models
model_math = RandomForestRegressor()
model_math.fit(X_train, y_math_train)

model_reading = RandomForestRegressor()
model_reading.fit(X_train, y_reading_train)

model_writing = RandomForestRegressor()
model_writing.fit(X_train, y_writing_train)

# Evaluate models
math_predictions = model_math.predict(X_test)
reading_predictions = model_reading.predict(X_test)
writing_predictions = model_writing.predict(X_test)

print("Math score RMSE:", mean_squared_error(y_math_test, math_predictions, squared=False))
print("Reading score RMSE:", mean_squared_error(y_reading_test, reading_predictions, squared=False))
print("Writing score RMSE:", mean_squared_error(y_writing_test, writing_predictions, squared=False))
