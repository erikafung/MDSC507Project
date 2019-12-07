#Code modified from
#https://datascienceplus.com/principal-component-analysis-pca-with-python/
#https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Initially read the combined clinical and RNA seq data set and transpose data (80 individuals)
#pd.read_csv("testPCA.csv", header=None, low_memory = False).T.to_csv('outputTransposeV3.csv', header=False, index=False)
Alz_data = pd.read_csv('outputTransposeV3.csv', low_memory = False)

#outputTransposeV3 csv needed to be altered to run fit() so Study column (String) deleted, avg ages inserted where NaN
df = pd.DataFrame(Alz_data, columns=Alz_data.keys())

n_samples, n_features = df.shape

#Scale values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
scaled_data = np.nan_to_num(scaled_data)

#Perform PCA
from sklearn.decomposition import PCA

#79 components used because
#ValueError: n_components must be between 0 and min(n_samples, n_features)=79 with svd_solver='full'
#error thrown if n_components is higher than 79
pca = PCA(n_components = 79, svd_solver = 'full')
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
principalDf = pd.DataFrame(data = x_pca)

explained_variance = pca.explained_variance_ratio_
print('Explained variance: ', explained_variance)
total_variance = sum(explained_variance)
print('Total variance explained by principal components: ', total_variance)
#Store principal components as CSV for use in random forest model
principalDf.to_csv (r'C:\Users\erikafung\Documents\MDSC\mdsc 507\PCAdataframeV2.csv', index = None, header=True)
