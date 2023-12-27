# Firstly, I followed the instructor on youtube link to write a script he has
# Then, I created this script to exclude the parts which I see as universal to be implemented
# in most machine learning applications.

import os

import numpy as np

import pandas as pd

import plotly.express as px

from sklearn.impute import SimpleImputer

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

import streamlit as st

def outlier_treatment(data_c):

    sorted(data_c)

    Q1,Q3 = np.percentile(data_c,[25,75])

    IQR = Q3-Q1

    lower_range = Q1-(1.5*IQR)
    upper_range = Q3+(1.5*IQR)

    return lower_range,upper_range

def description_panel(df):

    st.dataframe(df)

    st.subheader("Statistical Description")
    df.describe().T

    st.subheader("Balance of Data")
    st.bar_chart(df.iloc[:,-1].value_counts())

    null_df = df.isnull().sum().to_frame().reset_index()
    null_df.columns = ["Columns","Counts"]

    p1,p2,p3 = st.columns([2,1,2])

    p1.subheader("Null Variables")
    p1.dataframe(null_df)

    p2.subheader("Imputation")
    cat_m = p2.radio("Categorical",["Mode","BackFill","ForwardFill"])
    num_m = p2.radio("Numerical",["Mode","Median"])

    p2.subheader("Feature Engineering")
    balance_problem = p2.checkbox("Over Sampling")
    outlier_problem = p2.checkbox("Clean Outlier")

    if p2.button("Data preprocessing"):

        cat_cols = df.iloc[:,:-1].select_dtypes(include="object").columns
        num_cols = df.iloc[:,:-1].select_dtypes(exclude="object").columns

        if cat_cols.size > 0:

            if cat_m == "Mode":
                imp_cat = SimpleImputer(missing_values=np.nan,strategy="most_frequent")
                df[cat_cols] = imp_cat.fit_transform(df[cat_cols])

            elif cat == "BackFill":
                df[cat_cols].fillna(method="backfill",inplace=True)

            elif cat == "ForwardFill":
                df[cat_cols].fillna(method="ffill",inplace=True)

        if num_cols.size>0:

            if num_m == "Mode":
                imp_num = SimpleImputer(missing_values=np.nan,strategy="most_frequent")
            elif num_m == "Median":
                imp_num = SimpleImputer(missing_values=np.nan,strategy="median")

            df[num_cols] = imp_num.fit_transform(df[num_cols])

        df.dropna(axis=0,inplace=True)

        if balance_problem:
            over_sample = RandomOverSampler()
            X = df.iloc[:,:-1]
            y = df.iloc[:,[-1]]

            X,y = over_sample.fit_resample(X,y)

            df = pd.concat([X,y],axis=1)

        if outlier_problem:

            for col in num_cols:
                lower_bound,upper_bound = outlier_treatment(df[col])
                df[col] = np.clip(df[col],a_min=lower_bound,a_max=upper_bound)

        null_df = df.isnull().sum().to_frame().reset_index()
        null_df.columns = ["Columns","Counts"]

        p3.subheader("Null Variables")
        p3.dataframe(null_df)

        st.subheader("Balance of Data")
        st.bar_chart(df.iloc[:,-1].value_counts())

        heatmap = px.imshow(df.select_dtypes(exclude="object").corr())
        st.plotly_chart(heatmap)
        st.dataframe(df)

        if os.path.exists("model.csv"):
            os.remove("model.csv")

        df.to_csv("model.csv",index=False)
