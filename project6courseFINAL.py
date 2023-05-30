import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt

# LOADING DATA 
file_path = input("input file path: ")                            # taking the file path and putting it in the value 'df'
df=pd.read_csv(file_path)           
global_sales_mean = df['Global_Sales'].mean()
df['Global_sales_mean'] = global_sales_mean

# DATA EXPLORATION

print("Shape of the dataset:", df.shape)                                # prints the shape of the dataset
print("Column names:", df.columns)                                      # prints the name of each column
print("data types of each coulmn:\n", df.dtypes)                        # prints the type of data 
print("sum of empty cells:\n",df.isnull().sum())                        # prints the sum of cells that has no value

# DATA CLEANING

year_mean= df['Year'].mean()
print(df['Year'].fillna(year_mean))                                     # fills the empty cells with the mean value of the column 
print(df['Publisher'].fillna("UNKNOWN PUBLISHER"))                      # fills the empty cells with this string: "UNKNOWN PUBLISHER"
df['Platform'] = df['Platform'].replace('2600', "UNKNOWN PLATFORM")     # replaces the wrong cells with the word: "UNKNOWN PLATFORM"


'''print(df.fillna(0)) "this should be filled with the mean or median"'''

# DATA VISUALIZATION

plot = df.groupby('Name')['Global_Sales'].sum().nlargest(50).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(plot.index, plot.values, color='red')
plt.xticks(rotation=90)
plt.title('video games global sales')
plt.xlabel('Video Game Title')
plt.ylabel('Global Sales (Millions)')
plt.show()
######################################################################################################3
top_10 = df.groupby('Name')['Global_Sales'].sum().nlargest(10).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(top_10.index, top_10.values, color='blue')
plt.xticks(rotation=90)
plt.title('Top 10 Video Games by Global Sales')
plt.xlabel('Video Game Title')
plt.ylabel('Global Sales (Millions)')
plt.show()

#  Descriptive Analysis
min_sales= df['Global_Sales'].min()
max_sales= df['Global_Sales'].max()
mean_sales = df['Global_Sales'].mean()
median_sales = df['Global_Sales'].median()
mode_sales = df['Global_Sales'].mode()

rounded_num = "{:.4f}".format(mean_sales)                              # limits the 'mean_sales' value to 4 numbers after the point

print("Min global sales: ",min_sales)
print("Mean global sales: ", rounded_num)
print("Median global sales: ", median_sales)
print("Mode global sales: ", mode_sales)
print("Max global sales: ",max_sales)

#  Interpretation and Findings

sns.barplot(x='Genre', y='Global_Sales', data=df)
plt.title('correlation between the genre and global sales')
plt.show()

#####################################################################################################################
# MACHINE LEARNING PART OF THE CODE
# Convert categorical variables to numerical using LabelEncoder
for col in df.columns:
    if df[col].dtype ==object:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Split the data into features and target variable
X = df.drop('Global_Sales', axis=1)
y = df['Global_Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'learning_rate': np.linspace(0.01, 0.2, 10),
    'max_depth': np.arange(2, 10),
    'max_leaf_nodes': [None] + list(np.arange(10, 100, 10)),
    'min_samples_leaf': [1, 2, 4],
    'l2_regularization': np.linspace(0.1, 0.5, 5),
    'max_bins': [50, 100, 150, 200]
}

# Create the HistGradientBoostingRegressor model
hist_model = HistGradientBoostingRegressor()

# Perform randomized search cross-validation to find the optimal hyperparameters
random_search = RandomizedSearchCV(hist_model, param_distributions=param_grid, n_iter=50, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', random_state=0)
random_search.fit(X_train_scaled, y_train)

# Print the best hyperparameters and mean squared error score
print(f"Best Hyperparameters: {random_search.best_params_}")
print(f"Train Mean Squared Error: {-random_search.best_score_}")

# Predict on the test set using the best model
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Calculate the mean squared error and R-squared for the test set
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the mean squared error and R-squared
print(f"Test Mean Squared Error: {mse}")
print(f"Test R-squared: {r2}")

# Create a scatter plot of the predicted vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Global Sales')
plt.show()
 
