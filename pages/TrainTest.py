import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title='DHV Train/Test', layout='wide')

st.markdown("<h1 style='text-align: center;'>DHV AI Startup: Preoperative Risk Prediction by Train/Test Data</h1>", unsafe_allow_html=True)
st.write("")

# Load the data from data.csv
data = pd.read_csv("data.csv")

# Separate features and target
columns_to_drop = ["caseid","icu_days", "death_inhosp","department","optype","dx","ane_type","preop_ecg","preop_pft",
 "preop_ph", "preop_hco3", "preop_be", "preop_pao2", "preop_paco2", "preop_sao2","intraop_epi","risk"]
X = data.drop(columns_to_drop, axis=1)
y = data["risk"]

# Separate the features (X) and the target variable (y)
data_processed = X
#y = data_processed["risk"]

# Perform label encoding for categorical features
label_encoder = LabelEncoder()
categorical_features = ["sex", "asa", "emop", "preop_htn", "preop_dm"]
for feature in categorical_features:
    X[feature] = label_encoder.fit_transform(X[feature])

# Handle Sex Feature: Encode the sex feature
sex_mapping = {'Male': 1, 'Female': 2}
X['sex'] = X['sex'].map(sex_mapping)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

import os

# Function to get the CSV file with the highest number in its name
def get_most_numbered_csv():
    folder_path = "user_input"
    files = os.listdir(folder_path)
    csv_files = [file for file in files if file.endswith(".csv")]
    if len(csv_files) == 0:
        return None
    csv_files.sort(key=lambda x: int(x.split(".")[0]))
    most_numbered_csv = csv_files[-1]
    return os.path.join(folder_path, most_numbered_csv)

# Get the path of the CSV file with the highest number
most_numbered_csv_path = get_most_numbered_csv()

if most_numbered_csv_path is not None:
    # Load the most recent CSV file
    df = pd.read_csv(most_numbered_csv_path)

    # Get the most recent user input data
    most_numbered_csv = df.iloc[-1]

     # Set the user input values
    age = most_numbered_csv["age"]
    sex = most_numbered_csv["sex"]
    bmi = most_numbered_csv["bmi"]
    asa = most_numbered_csv["asa"]
    emop = most_numbered_csv["emop"]
    preop_htn = most_numbered_csv["preop_htn"]
    preop_dm = most_numbered_csv["preop_dm"]
    preop_hb = most_numbered_csv["preop_hb"]
    preop_plt = most_numbered_csv["preop_plt"]
    preop_pt = most_numbered_csv["preop_pt"]
    preop_aptt = most_numbered_csv["preop_aptt"]
    preop_na = most_numbered_csv["preop_na"]
    preop_k = most_numbered_csv["preop_k"]
    preop_gluc = most_numbered_csv["preop_gluc"]
    preop_alb = most_numbered_csv["preop_alb"]
    preop_ast = most_numbered_csv["preop_ast"]
    preop_alt = most_numbered_csv["preop_alt"]
    preop_bun = most_numbered_csv["preop_bun"]
    preop_cr = most_numbered_csv["preop_cr"]

    # Create a DataFrame for user input
    most_numbered_csv = pd.DataFrame([[age, sex, bmi, asa, emop, preop_htn, preop_dm, preop_hb, preop_plt, preop_pt,
                                       preop_aptt, preop_na, preop_k, preop_gluc, preop_alb, preop_ast, preop_alt,
                                       preop_bun, preop_cr]], columns=X.columns)
  
    

st.dataframe(df)

# Predict the risk using the trained classifier
risk_prediction = clf.predict(most_numbered_csv)

st.title("3. Random Forest")
st.markdown("<h1 style='color: black;'>Risk Prediction (1 = low , 2-3 = moderate, 4 = high risk):</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='color: red;'>{}</h1>".format(risk_prediction[0]), unsafe_allow_html=True)
#st.markdown("<hr style='border: 1px solid black;'>", unsafe_allow_html=True)
st.markdown("<h2 style='color: black;'>Accuracy: {:.2f}</h2>".format(accuracy), unsafe_allow_html=True)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer

# Combine the user input with the training data
combined_data = pd.concat([X, most_numbered_csv], ignore_index=True)

# Perform imputation for missing values using KNNImputer
imputer = KNNImputer(n_neighbors=5)
imputed_combined_data = pd.DataFrame(imputer.fit_transform(combined_data), columns=combined_data.columns)

# Get the user input features (last row in the imputed_combined_data)
user_input_features = imputed_combined_data.iloc[[-1]]

# Split the data into training and testing sets
X_train = imputed_combined_data[:-1]
y_train = y

# Train the KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict the risk for the user input
risk_prediction = knn.predict(user_input_features)

# Map the predicted risk value to the corresponding label
risk_mapping_reverse = {1: "1", 2: "2,3", 3: "4"}
predicted_risk_label = risk_mapping_reverse[risk_prediction[0]]

# Calculate accuracy and generate classification report
y_train_pred = knn.predict(X_train)
accuracy = accuracy_score(y_train, y_train_pred)
classification_rep = classification_report(y_train, y_train_pred)

# Display accuracy and classification report
#st.write("Accuracy:", accuracy)
st.markdown("<hr style='border: 1px solid black;'>", unsafe_allow_html=True)
st.write("")
st.write("")
st.title("4. K-Nearest Neighbors")

# Display the predicted risk label
#st.markdown("<h1 style='color: black;'>Risk Prediction (1 = low , 2-3 = moderate, 4 = high risk):</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='color: red;'>{}</h1>".format(predicted_risk_label[0]), unsafe_allow_html=True)
st.markdown("<h2 style='color: black;'>Accuracy: {:.2f}</h2>".format(accuracy), unsafe_allow_html=True)
st.title("Classification Report:")
st.write(classification_rep)