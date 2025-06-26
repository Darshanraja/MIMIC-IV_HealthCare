import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, chi2_contingency
import seaborn as sns
import streamlit as st

# Load datasets
demo_first = pd.read_csv('data/csv/demo_all.csv')
labels_first = pd.read_csv('data/csv/labels.csv')
dynamic_first = pd.read_csv('data/csv/dynamic_all.csv')
static_first = pd.read_csv('data/csv/static_all.csv')

demo_last = pd.read_csv('data/csv10/demo_all-last10.csv')
labels_last = pd.read_csv('data/csv10/labels-last10.csv')
dynamic_last = pd.read_csv('data/csv10/dynamic_all-last10.csv')
static_last = pd.read_csv('data/csv10/static_all-last10.csv')

# Merge datasets
def merge_datasets(labels, demo, dynamic, static):
    data = labels.merge(demo, left_on='stay_id', right_on='hid', how='inner')
    data = data.merge(dynamic, left_on='stay_id', right_on='ids', how='inner')
    data = data.merge(static, left_on='stay_id', right_on='ids', how='inner')
    return data

data_first = merge_datasets(labels_first, demo_first, dynamic_first, static_first)
data_last = merge_datasets(labels_last, demo_last, dynamic_last, static_last)

# Select relevant columns
columns_to_keep = ['Age', 'gender', 'ethnicity']
data_first = data_first[columns_to_keep]
data_last = data_last[columns_to_keep]

# Exclude rows with NaN in ethnicity
data_first = data_first.dropna(subset=['ethnicity'])
data_last = data_last.dropna(subset=['ethnicity'])

# Data preprocessing
# Standardize Age across both datasets
combined_mean = pd.concat([data_first['Age'], data_last['Age']]).mean()
combined_std = pd.concat([data_first['Age'], data_last['Age']]).std()
data_first['Age'] = (data_first['Age'] - combined_mean) / combined_std
data_last['Age'] = (data_last['Age'] - combined_mean) / combined_std

# Label encode Gender and Ethnicity consistently
for col in ['gender', 'ethnicity']:
    unique_vals = list(set(data_first[col].unique()).union(set(data_last[col].unique())))
    encoding = {val: idx for idx, val in enumerate(unique_vals)}
    data_first[col] = data_first[col].map(encoding)
    data_last[col] = data_last[col].map(encoding)

# Perform data distribution shift analysis
def measure_shift(data1, data2):
    shift_results = {}

    # Numerical Feature - Age (KS Test)
    ks_stat, ks_pval = ks_2samp(data1['Age'], data2['Age'])
    shift_results['Age'] = {"KS Statistic": ks_stat, "p-value": ks_pval}

    # Categorical Features - Gender and Ethnicity (Chi-Square Test)
    for col in ['gender', 'ethnicity']:
        contingency_table = pd.crosstab(data1[col], data2[col])
        chi2_stat, chi2_pval, _, _ = chi2_contingency(contingency_table)
        shift_results[col] = {"Chi-Square Statistic": chi2_stat, "p-value": chi2_pval}

    return shift_results

# Measure data shift
shift_results = measure_shift(data_first, data_last)

# Streamlit visualization
st.title("Data Distribution Shift Analysis")

# Display shift results
st.subheader("Shift Results")
for feature, stats in shift_results.items():
    st.write(f"**Feature:** {feature}")
    for metric, value in stats.items():
        st.write(f"  {metric}: {value:.4f}")

# Visualize distributions
st.subheader("Distributions")

# Define mappings for human-readable labels
gender_mapping = {0: "Male", 1: "Female"}  # Adjust based on your data
ethnicity_mapping = {0: "White", 1: "Black", 2: "Hispanic", 3: "Asian"}  # Adjust based on your data

# Age Distribution with Values
st.write("**Age Distribution**")
fig, ax = plt.subplots(figsize=(8, 4))
sns.kdeplot(data_first['Age'], shade=True, label="First 10%", ax=ax)
sns.kdeplot(data_last['Age'], shade=True, label="Last 10%", ax=ax)
ax.set_title("Density Plot for Age")
ax.set_xlabel("Standardized Age")
ax.legend()

# Overlay actual Age values
first_age_mean = combined_mean + combined_std * data_first['Age'].mean()
last_age_mean = combined_mean + combined_std * data_last['Age'].mean()
ax.axvline(data_first['Age'].mean(), color="blue", linestyle="--", label=f"First 10% Mean: {first_age_mean:.2f}")
ax.axvline(data_last['Age'].mean(), color="orange", linestyle="--", label=f"Last 10% Mean: {last_age_mean:.2f}")
ax.legend()
st.pyplot(fig)

# Gender Distribution
st.write("**Gender Distribution**")
fig, ax = plt.subplots(figsize=(8, 4))
gender_counts = pd.concat([
    data_first['gender'].value_counts(normalize=True).rename("First 10%"),
    data_last['gender'].value_counts(normalize=True).rename("Last 10%"),
], axis=1)
gender_counts.index = gender_counts.index.map(gender_mapping)  # Map to human-readable labels
gender_counts.plot(kind='bar', stacked=False, ax=ax)
ax.set_title("Gender Distribution Comparison")
ax.set_xlabel("Gender")
ax.set_ylabel("Proportion")
st.pyplot(fig)

# Ethnicity Distribution
st.write("**Ethnicity Distribution**")
fig, ax = plt.subplots(figsize=(8, 4))
ethnicity_counts = pd.concat([
    data_first['ethnicity'].value_counts(normalize=True).rename("First 10%"),
    data_last['ethnicity'].value_counts(normalize=True).rename("Last 10%"),
], axis=1)
ethnicity_counts.index = ethnicity_counts.index.map(ethnicity_mapping)  # Map to human-readable labels
ethnicity_counts.plot(kind='bar', stacked=False, ax=ax)
ax.set_title("Ethnicity Distribution Comparison")
ax.set_xlabel("Ethnicity")
ax.set_ylabel("Proportion")
st.pyplot(fig)
