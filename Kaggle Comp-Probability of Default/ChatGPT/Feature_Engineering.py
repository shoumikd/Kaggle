import pandas as pd
import numpy as np
#from llama_index import SimpleDocument, GPTListIndex
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ChatGPT.FeatureExtractor import FeatureExtractor

# File Constants
file_path = "..\\Kaggle Comp-Probability of Default\\ChatGPT\\processed_data\\cleaned_predicted.csv"
target_column='loan_status'

numerical_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
                      'cb_person_cred_hist_length']
categorical_features_for_ordinal_encoding = ['cb_person_default_on_file']
categorical_features_for_one_hot_encoding = ['person_home_ownership', 'loan_intent', 'loan_grade']


def readFile(file_path):
    print("Reading File from Absolute Path:",os.path.abspath(file_path))

    try:
        with open(file_path) as f:
            df = pd.read_csv(file_path)
            print("Dataframe Created from Cleaned File")
            return df
    except FileNotFoundError:
        raise FileNotFoundError('The file does not exist.')
    

def dataPreprosessing(df, numerical_required_features=numerical_features, categorical_required_features_for_ordinal_encoding=categorical_features_for_ordinal_encoding, 
                      categorical_required_features_for_one_hot_encoding=categorical_features_for_one_hot_encoding):
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_required_features),
        ('cat_label_encoding', OrdinalEncoder(), categorical_required_features_for_ordinal_encoding),
        ('cat_one_hot_encoding', OneHotEncoder(drop='first', sparse_output=False), categorical_required_features_for_one_hot_encoding)
    ])

    # Create the full pipeline
    pipeline = Pipeline([
        ('feature_extraction', FeatureExtractor()),
        ('preprocessing', preprocessor)
    ])

    X_transformed= pipeline.fit_transform(df)
    X_transformed_df = pd.DataFrame(X_transformed, columns=[*numerical_required_features,*categorical_required_features_for_ordinal_encoding,
        *pipeline.named_steps['preprocessing'].named_transformers_['cat_one_hot_encoding'].get_feature_names_out(categorical_required_features_for_one_hot_encoding)
    ])

    return X_transformed_df



# Load and process data
df=readFile(file_path)
X = df.drop(columns=[target_column])
y = df[target_column]
X_transformed_df=dataPreprosessing(X)


