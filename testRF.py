#Code modified from
#https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
#Run after testPCA.py
import pandas as pd

#read in csv of PCA components with column added for cogdx values for the 80 individuals
features = pd.read_csv('PCAdataframeV2.csv')


import numpy as np

# Labels are the values we want to predict
labels = np.array(features['cogdx'])

# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('cogdx', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
print('Mean Error:', round(np.mean(errors), 2))

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
