import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

housing = pd.read_csv('CaliforniaHousingData.csv')
housing.info()

#summary of the dataset housing, showing descriptive statistics for all columns, 
#regardless of their data type (numerical, categorical, etc.)
housing.describe(include='all')

#calculating the total number of null values in each column
housing.isna().sum()

#removing all rows containing null (missing) values from the housing DataFrame
housing.dropna(inplace=True)

#histogram to visualize the distribution of the median_house_value column in the housing DataFrame
plt.figure(figsize=(10,8))
housing['median_house_value'].hist(edgecolor='black',grid=False)

#This line displays the count of occurrences for each unique value in the `ocean_proximity` 
#column of the `housing` DataFrame, providing a summary of the distribution of categories in that column.
housing['ocean_proximity'].value_counts()

#converting the categorical column housing['ocean_proximity'] into a one-hot encoded DataFrame. 
#Each unique value in the column becomes a separate binary column,
pd.get_dummies(housing['ocean_proximity'],dtype=int)

housing_copy = housing.copy()

#After execution, the housing DataFrame will contain the original columns plus new binary 
#columns corresponding to each unique category in the ocean_proximity column.
housing = housing.join(pd.get_dummies(housing['ocean_proximity'],dtype=int))

housing.drop(columns = ['ocean_proximity'], inplace=True)

housing

#generates histograms for all numerical columns in the housing DataFrame, arranging them in a grid layout for 
#easier visualizationsns.scatterplot(data=housing, x='latitude', y='longitude', hue='median_house_value', palette = 'cividis')
plt.figure(figsize=(14,8))
sns.scatterplot(data=housing, x='latitude', y='longitude', hue='median_house_value', palette = 'cividis')

#generating histograms for all numerical columns in the housing DataFrame, arranging them in a grid layout for easier visualization
housing.hist(figsize=(14,8))
plt.tight_layout()

#Creating a checkpoint
housing_checkpoint2 = housing.copy()

housing = housing_checkpoint2

import numpy as np

#applying a logarithmic transformation to certain numerical columns in the housing DataFrame
housing['total_bedrooms'] = np.log(housing['total_bedrooms']+1)
housing['total_rooms'] = np.log(housing['total_rooms']+1)
housing['populations'] = np.log(housing['population']+1)
housing['households'] = np.log(housing['households']+1)

housing.hist(figsize=(14,8))
plt.tight_layout()

#Feature Engineering
housing

#creates a new column, avg_rooms_per_house, in the housing DataFrame by calculating 
#the average number of rooms per household
housing['avg_rooms_per_house'] = housing['total_rooms'] / housing['households']

housing

#generating a heatmap to visualize the correlation matrix of the housing DataFrame.
plt.figure(figsize=(14,8))
sns.heatmap(housing.corr(), annot=True, cmap = 'viridis')

#calculating the ratio of bedrooms to total rooms for each row in the housing DataFrame and stores it in a new column called bedroom_ratio.
housing['bedroom_ratio'] = housing['total_bedrooms']/housing['total_rooms']

housing_copy3 = housing.copy()

#to remove multicollinearity
housing.drop(columns = ['total_rooms', 'total_bedrooms'], inplace=True)

plt.figure(figsize=(14,8))
sns.heatmap(housing.corr(), annot=True, cmap = 'viridis')

#because most correlation values associated with population are less than -0.8
housing.drop(columns = ['population'], inplace=True)

plt.figure(figsize=(14,8))
sns.heatmap(housing.corr(), annot=True, cmap = 'viridis')

#creates a new DataFrame X by dropping the median_house_value column from the housing DataFrame
X = housing.drop(columns = ['median_house_value'])

y = housing['median_house_value']

X,y

#This line splits the dataset (`X`, `y`) into training and testing subsets, allocating 20% of 
#the data to testing and ensuring reproducibility with `random_state=42`.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#This line creates an instance of the `LinearRegression` class, initializing a linear regression model to fit data and make predictions.
model = LinearRegression()

#This line trains the linear regression model by fitting it to the training data (`X_train` for features 
#and `y_train` for target values), allowing the model to learn the relationship between the inputs and outputs.
model.fit(X_train, y_train)

#generating a summary of descriptive statistics for the training dataset X_train
X_train.describe()

#This line evaluates the performance of the trained model on the testing data by calculating the coefficient of 
#determination score, which measures how well the model predicts the target values in  y_test based on X_test.
model.score(X_test, y_test)

#This line evaluates the performance of the trained model on the testing data by calculating the 
#coefficient of determination (\( R^2 \)) score, which measures how well the model predicts the target 
#values in y_test based on X_test.
y_pred = model.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_pred)
r2

mse = mean_squared_error(y_test, y_pred)

np.sqrt(mse)

housing.describe()

from sklearn.ensemble import RandomForestRegressor

model_forest = RandomForestRegressor()

model_forest.fit(X_train, y_train)

RandomForestRegressor()

model_forest.score(X_test, y_test)

y_pred_for = model_forest.predict(X_test)
mse_for = mean_squared_error(y_test, y_pred_for)
print("Mean Squared Error:",  mse_for)

np.sqrt(2436337250.0554104)

feature_names = X_train.columns
feature_importance = model_forest.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10,6))

# Horizontal bar plot
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')

# Add title and labels
plt.title("Feature Importance in Model", fontsize=16)
plt.xlabel("Importance Score", fontsize=14)
plt.ylabel("Features", fontsize=14)

# Display the plot
plt.show()