#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


# In[2]:


def dataPreprocessing(df):
    
    # Handling Outliers
    cr_copy=df.copy()
    age_filter=cr_copy['person_age']<=68
    cr_age_filtered=cr_copy[age_filter]
    emp_length_filter=cr_age_filtered['person_emp_length']<=47
    cr_emp_length_filtered=cr_age_filtered[emp_length_filter]
    cr_emp_length_filtered.reset_index(drop=True,inplace=True)

    #Simple Imputation of Numerical Data
    cr_int_rate_filled=cr_emp_length_filtered.copy()
    cr_int_rate_filled=cr_int_rate_filled.fillna({'loan_int_rate':cr_int_rate_filled['loan_int_rate'].mean()})
    cr_loan_grade_rmvd=cr_int_rate_filled.drop('loan_grade',axis=1)

    
    #Encoding Of Categorical Data
    cr_loan_cat_treated=cr_loan_grade_rmvd.copy()
    categorical_columns=cr_loan_cat_treated.select_dtypes(include=['object']).columns.to_list
    one_hot_encoding_columns=['person_home_ownership', 'loan_intent']
    binary_encoding_columns=['cb_person_default_on_file']
    
    #1. One Hot Encoding of Categorical Columns
    ohe=OneHotEncoder(sparse_output=False)
    one_hot_encoded=ohe.fit_transform(cr_loan_cat_treated[one_hot_encoding_columns])
    cr_ohe=pd.DataFrame(one_hot_encoded, columns=ohe.get_feature_names_out(one_hot_encoding_columns))
    cr_loan_cat_treated = pd.concat([cr_loan_cat_treated, cr_ohe], axis=1)

    #2. Label Encoding/ Binary Encoding of Binary Columns
    cr_loan_bin_treated=cr_loan_cat_treated.copy()
    cr_loan_bin_treated[binary_encoding_columns]=np.where((cr_loan_bin_treated[binary_encoding_columns])=='Y',1,0)
    

    #Standardization of Numerical Data
    cr_loan_scaled=cr_loan_bin_treated.copy()
    scale_columns=['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
    cr_loan_scaled[scale_columns] = StandardScaler().fit_transform(cr_loan_scaled[scale_columns])

    
    #Drop Non Required Columns
    dropColumns=['person_home_ownership','loan_intent','person_home_ownership_OTHER','loan_intent_DEBTCONSOLIDATION']
    cr_loan_processed=cr_loan_scaled.copy()
    cr_loan_processed= cr_loan_processed.drop(dropColumns, axis=1)

    #Handling Under-Samplimg
    smote=SMOTE()
    independent=cr_loan_processed.drop('loan_status', axis=1)
    dependent=cr_loan_processed['loan_status']
    x_smote, y_smote=smote.fit_resample(independent,dependent)

    return cr_emp_length_filtered, x_smote, y_smote


# In[3]:


df=pd.read_csv('./Training/data/credit_risk_dataset.csv')

cr_emp_length_filtered, x_smote, y_smote=dataPreprocessing(df)


# In[4]:


cr_emp_length_filtered.shape


# In[5]:


x_smote.shape


# In[6]:


y_smote.shape


# In[ ]:




