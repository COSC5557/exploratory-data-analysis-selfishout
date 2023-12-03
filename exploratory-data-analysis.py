import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target


print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())



df.hist(bins=10, figsize=(20, 15))
plt.show()


plt.figure(figsize=(20, 10))
df.boxplot()
plt.xticks(rotation=90)
plt.show()

# Correlation Matrix
corr_matrix = df.corr()
plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, annot=True, fmt='.1f')
plt.show()


sns.countplot(x='target', data=df)
plt.show()

# Standardizing the Data
features = df.drop('target', axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Displaying scaled features
print("scaled")
print(pd.DataFrame(features_scaled, columns=data.feature_names).head())



# Preprocessing Part

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessing_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  
    ('scaler', StandardScaler()),                
    ('poly', PolynomialFeatures(degree=2)),      
    ('select', SelectKBest(f_classif, k=10))     
])


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessing_pipeline),
    ('classifier', LogisticRegression())
])


pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)


print("Model Accuracy:", accuracy_score(y_test, y_pred))





# Another version 


# data = load_breast_cancer()
# df = pd.DataFrame(data.data, columns=data.feature_names)
# df['target'] = data.target


# basic_stats = df.describe()
# print(basic_stats)

# # Missing values
# print(df.isnull().sum())


# sns.countplot(x='target', data=df)
# plt.show()

# # Detect outliers using Z-score for 'mean radius'
# z_scores = np.abs(stats.zscore(df['mean radius']))
# outliers = np.where(z_scores > 3)
# print("outliers  ****************************")
# print(outliers)

# # Normalization/Standardization
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(df.drop('target', axis=1))
# df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])


# df_scaled['area_perimeter_ratio'] = df_scaled['mean area'] / df_scaled['mean perimeter']

# selected_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
# plt.figure(figsize=(15, 10))
# for i, feature in enumerate(selected_features, 1):
#     plt.subplot(2, 2, i)
#     sns.histplot(df[feature], kde=True, bins=20)
#     plt.title(f'Distribution of {feature}')
#     plt.xlabel(feature)
#     plt.ylabel('Count')


# plt.figure(figsize=(15, 15))
# corr_matrix = df.corr()
# sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
# plt.title('Correlation Matrix of Breast Cancer Features')
# plt.show()



