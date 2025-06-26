MIMIC-IV HealthCare Project

This repository presents an end-to-end data analysis and fairness evaluation pipeline using the MIMIC-IV dataset, focusing on healthcare prediction tasks. The work emphasizes data preprocessing, feature selection, model interpretation, and bias mitigation techniques.

📊 Project Overview

The goal of this project is to explore patient data from MIMIC-IV (via PhysioNet), clean and transform it for predictive tasks, and evaluate model fairness across different demographic groups.

⚠️ Note: Due to licensing constraints, raw MIMIC-IV data is not included. You must request access directly from PhysioNet.
🗂️ Repository Contents

Final_Project.ipynb – Complete analysis pipeline: from loading and cleaning MIMIC-IV data to feature selection, model building, and fairness evaluation.
fairness_app.py – Script to evaluate model bias using metrics like demographic parity, equal opportunity, and fairness-aware visualizations.
2nd_project.ipynb & Final_(1).ipynb – Earlier exploratory and comparative notebooks.
README.md – This file.
✅ Key Features

📥 Data Cleaning: Null value handling, patient filtering, ICU stay selection.
⚙️ Feature Selection: Utilized LIME and other explainability tools for identifying impactful features.
🧠 Modeling: Logistic regression and ensemble classifiers trained for outcome prediction.
📉 Fairness Analysis: Compared model performance across sensitive attributes (e.g., race, gender).
📌 Ethical Focus: Included fairness metrics in model evaluation to mitigate health disparities.
🧪 Requirements

Python 3.8+
pandas, numpy, scikit-learn
lime
fairlearn
matplotlib / seaborn
Install all dependencies via:

pip install -r requirements.txt
(If requirements.txt is not provided, install manually using pip.)

🔐 Data Access

To reproduce results, you must:

Create an account on PhysioNet.
Apply for access to the MIMIC-IV dataset.
Download and store the data locally as specified in the notebook paths.
👤 Author

Darshan Raja
Master's Student, UNCC Charlotte
