#Code modified from
#https://www.geeksforgeeks.org/principal-component-analysis-with-python/
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Initially read the combined clinical and RNA seq data set and transpose data (80 individuals)
#pd.read_csv("testPCA.csv", header=None, low_memory = False).T.to_csv('outputTranspose3.csv', header=False, index=False)
Alz_data = pd.read_csv('outputTransposeV3.csv', low_memory = False)

#16380 is the number of columns in the table
X = Alz_data.iloc[:, 0:16380].values
y = Alz_data.iloc[:,16380].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Scale values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Perform PCA
from sklearn.decomposition import PCA

#63 components used because
#ValueError: n_components must be between 0 and min(n_samples, n_features)=63 with svd_solver='full'
#error thrown if n_components is higher than 63
pca = PCA(n_components = 63, svd_solver = 'full')
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)
y_train = np.nan_to_num(y_train)
y_test = np.nan_to_num(y_test)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_
print('Explained variance: ', explained_variance)
total_variance = sum(explained_variance)
print('Total variance explained by principal components: ', total_variance)

from sklearn import preprocessing
from sklearn import utils

lab_enc = preprocessing.LabelEncoder()
y_train_training_scores_encoded = lab_enc.fit_transform(y_train)
y_test_training_scores_encoded = lab_enc.fit_transform(y_test)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train)

# Use the forest's predict method on the test data
predictions = rf.predict(X_test)
# Calculate the absolute errors
errors = abs(predictions - y_test)
print('Mean Error:', round(np.mean(errors), 2))

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
