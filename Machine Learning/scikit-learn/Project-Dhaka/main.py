from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv("housing.csv")

# 2. Create a stratified test set
df["income_cat"] = pd.cut(df["median_income"], bins=[0, 1.5, 3, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = df.loc[test_index].drop("income_cat", axis=1)

# Will work on the copy of the training data
housing = strat_train_set.copy()

# 3. Separate features and labels
housing = housing.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# 4. Separate numerical and categorical columns
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]

# 5. Create the pipeline for numerical columns
# For numerical columns
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler())
])

# For categorical columns
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Construct the full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)
print(housing_prepared.shape)

# 7. Train different  models

# Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
print(f'Linear Regression RMSE: {lin_rmse}')

# cross-validation
lin_rmse_cross = np.sqrt(-cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10))
print(pd.Series(lin_rmse_cross).describe())

# Decision Tree Regression Model
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared, housing_labels)
dec_preds = dec_reg.predict(housing_prepared)
dec_rmse = root_mean_squared_error(housing_labels, dec_preds)
print(f'Decision Tree Regression RMSE: {np.sqrt((dec_rmse))}')

# cross-validation
tree_rmse_cross = np.sqrt(-cross_val_score(dec_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10))
print(pd.Series(tree_rmse_cross).describe())

# Random Forest Regression Model
rf_reg = RandomForestRegressor()
rf_reg.fit(housing_prepared, housing_labels)
rf_preds = rf_reg.predict(housing_prepared)
rf_rmse = root_mean_squared_error(housing_labels, rf_preds)
print(f'Random Forest Regression RMSE: {rf_rmse}')

# cross-validation
rf_rmse_cross = np.sqrt(-cross_val_score(rf_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10))
print(pd.Series(rf_rmse_cross).describe())