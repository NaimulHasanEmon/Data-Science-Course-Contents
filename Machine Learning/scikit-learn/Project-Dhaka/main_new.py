import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "housing_model.pkl"
PIPELINE_FILE = "housing_pipeline.pkl"

def build_pipeline(num_attributes, cat_attributes):
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
        ("num", num_pipeline, num_attributes),
        ("cat", cat_pipeline, cat_attributes)
    ])
    
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # Model does not exit, proceed with model training
    # Load the dataset
    df = pd.read_csv("housing.csv")

    # Create a stratified test set
    df["income_cat"] = pd.cut(df["median_income"], bins=[0, 1.5, 3, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(df, df["income_cat"]):
        df.loc[test_index].drop("income_cat", axis=1).to_csv("input.csv", index=False)
        housing = df.loc[train_index].drop("income_cat", axis=1)

    # Separate features and labels
    housing_features = housing.drop("median_house_value", axis=1)
    housing_labels = housing["median_house_value"].copy()

    # Separate numerical and categorical columns
    num_attributes = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attributes = ["ocean_proximity"]
    
    pipeline = build_pipeline(num_attributes, cat_attributes)
    housing_prepared = pipeline.fit_transform(housing_features)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)
    
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model and pipeline saved.")

else:
    # Inference
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    
    input_data = pd.read_csv("input.csv")
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data["median_house_value"] = predictions
    input_data.to_csv("output.csv", index=False)
    print("Inference complete! Predictions saved to output.csv file.")