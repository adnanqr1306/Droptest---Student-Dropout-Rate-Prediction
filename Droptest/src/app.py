import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score
from data_processing import get_project_root, load_data, preprocess_data

def calculate_specificity_sensitivity(y_true, y_pred):
    """Calculate specificity and sensitivity from confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    return specificity, sensitivity

# Load data
project_root = get_project_root()

# Load and preprocess full dataset
df_full = load_data()
df_full = preprocess_data(df_full)
X_full = df_full.drop('Dropped_Out', axis=1)
y_full = df_full['Dropped_Out']
expected_features = list(X_full.columns)

# Load test set for model comparison
test_set_path = Path(project_root) / 'data' / 'processed' / 'test_set.csv'
test_df = pd.read_csv(test_set_path)
X_test = test_df.drop('Dropped_Out', axis=1)
y_test = test_df['Dropped_Out']

# Load all models for comparison
models_dir = Path(project_root) / 'all_models'
model_files = [f for f in models_dir.glob('*.pkl') if f.is_file()]

# Calculate metrics for all models on TEST SET
results = []
for model_file in model_files:
    model = joblib.load(model_file)
    model_name = model_file.stem.replace('_', ' ').title()
    y_pred_test = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred_test)
    specificity, sensitivity = calculate_specificity_sensitivity(y_test, y_pred_test)
    
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Specificity': specificity,
        'Sensitivity': sensitivity,
        'F1 Score': f1_score(y_test, y_pred_test)
    })

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)

# Load best model
best_model_path = Path(project_root) / 'best_model' / 'best_model.pkl'
best_model = joblib.load(best_model_path)
best_model_name = type(best_model.named_steps['classifier']).__name__ if hasattr(best_model, 'named_steps') else type(best_model).__name__

# Get predictions for FULL DATASET
y_pred_full = best_model.predict(X_full)
y_proba_full = best_model.predict_proba(X_full)[:, 1]

# Streamlit UI setup
st.set_page_config(page_title="Droptest", layout="wide")

# Custom CSS styling (Updated)
st.markdown(
    """
<style>
/* Main app styling */
.stApp {
    background-color: #f5f7fb;
    font-family: 'Inter', sans-serif;
    color: #2d3436 !important;
}

/* Header/Deployment Bar Styling */
[data-testid="stHeader"] {
    background-color: #ffffff !important;
    border-bottom: 1px solid #e1e4eb !important;
}

.stDeployButton {
    background-color: #f0f2f6 !important;
    color: #2d3436 !important;
    border-radius: 6px;
    padding: 8px 16px !important;
    margin: 4px !important;
    border: 1px solid #e1e4eb !important;
}

#MainMenu {
    background-color: #f0f2f6 !important;
    border-radius: 6px !important;
    padding: 4px !important;
    margin: 4px !important;
}

#MainMenu button {
    color: #2d3436 !important;
}

.st-c0 {
    color: #2d3436 !important;
    background-color: white !important;
}

.st-c0:hover {
    background-color: #f5f7fb !important;
    color: #2d3436 !important;
}

/* Table styling */
div[data-testid="stDataFrame-container"] table {
    background-color: #f5f7fb !important;
    width: 100% !important;
    border: 1px solid #e1e4eb !important;
}

div[data-testid="stDataFrame-container"] th {
    background-color: #e1e4eb !important;
    color: #2d3436 !important;
    font-weight: 600 !important;
    padding: 12px !important;
    border-bottom: 2px solid #d0d4dd !important;
}

div[data-testid="stDataFrame-container"] td {
    background-color: #ffffff !important;
    color: #2d3436 !important;
    padding: 10px !important;
    border-bottom: 1px solid #e1e4eb !important;
}

div[data-testid="stDataFrame-container"] tr:hover td {
    background-color: #f0f2f6 !important;
}

div[data-testid="stDataFrame-container"] > div {
    height: auto !important;
}

div[data-testid="stDataFrame-container"] thead tr th:first-child,
div[data-testid="stDataFrame-container"] tbody tr td:first-child {
    display: none;
}

/* Text elements */
p, div, span, label, h1, h2, h3, h4, h5, h6 {
    color: #2d3436 !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 2px solid #e1e4eb;
}

/* Selectbox styling */
.stSelectbox [data-baseweb="select"] {
    background-color: #4A90E2 !important;
    border-radius: 4px !important;
    border: none !important;
}

.stSelectbox [data-baseweb="select"] div {
    color: #ffffff !important;
}

.stSelectbox [aria-live="polite"] div {
    color: #ffffff !important;
}

.stSelectbox [role="listbox"] div:hover {
    background-color: #357ABD !important;
    color: #ffffff !important;
}

/* Widget labels */
.stRadio label, .stSelectbox label, .stSlider label {
    color: #2d3436 !important;
    font-weight: 500;
}

/* Buttons */
.stButton > button {
    background-color: #4CAF50 !important;
    color: white !important;
    border-radius: 8px;
    padding: 12px 28px;
    transition: transform 0.2s;
}

.stButton > button:hover {
    transform: scale(1.05);
}

/* Progress bar */
.stProgress > div > div > div {
    background-color: #4CAF50 !important;
}

/* Custom cards */
.custom-card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

/* Metric boxes */
.metric-box {
    padding: 20px;
    border-radius: 8px;
    text-align: center;
    margin: 10px 0;
}

/* Plot containers */
.stPlotContainer {
    color: #2d3436 !important;
}
</style>
    """,
    unsafe_allow_html=True
)

# Main header
st.markdown(
    "<h1 style='text-align: center; margin-bottom: 30px;'>🎓 Droptest - Student Dropout Predictor</h1>",
    unsafe_allow_html=True
)

# Model Comparison Table (Test Set)
st.markdown("<h2 style='text-align: center; margin: 2rem 0 1rem;'>📊 Model Performance Comparison (Test Set)</h2>", unsafe_allow_html=True)
st.markdown("<div class='custom-card' style='padding: 0; background-color: #f5f7fb; box-shadow: none;'>", unsafe_allow_html=True)

# Create styled dataframe
styled_df = results_df.style\
    .format({
        'Accuracy': '{:.2%}',
        'Specificity': '{:.2%}',
        'Sensitivity': '{:.2%}',
        'F1 Score': '{:.2%}'
    })\
    .background_gradient(cmap='Blues', subset=['Accuracy'])\
    .set_properties(**{
        'padding': '8px 12px',
        'border': 'none',
        'font-size': '14px'
    })

st.dataframe(
    styled_df,
    height=(35 * len(results_df) + 40),  # Dynamic height based on rows
    use_container_width=True
)
st.markdown("</div>", unsafe_allow_html=True)

# Best Model Performance Section (Full Dataset)
st.markdown(f"<h2 style='text-align: center; margin-top: 40px;'>Best Model Performance: {best_model_name} (Full Dataset)</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Confusion Matrix</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        confusion_matrix(y_full, y_pred_full),
        annot=True,
        fmt='d',
        cmap='YlGn',
        linewidths=0.5,
        ax=ax
    )
    ax.set_xlabel('Predicted Labels', color='#2d3436')
    ax.set_ylabel('Actual Labels', color='#2d3436')
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>ROC Curve</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_full, y_proba_full)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='#4CAF50', lw=2, 
            label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='#666666', linestyle='--')
    ax.set_xlabel('False Positive Rate', color='#2d3436')
    ax.set_ylabel('True Positive Rate', color='#2d3436')
    ax.legend(loc='lower right')
    
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("📝 Student Information")
user_input = {}

# Demographic Information
st.sidebar.subheader("Demographics")
user_input['School'] = 1 if st.sidebar.radio('School', ['MS', 'GP']) == 'MS' else 0
user_input['Gender'] = 1 if st.sidebar.radio('Gender', ['Male', 'Female']) == 'Male' else 0
user_input['Address'] = 1 if st.sidebar.radio('Address', ['Urban', 'Rural']) == 'Urban' else 0

# Family Information
st.sidebar.subheader("Family Background")
user_input['Family_Size'] = 1 if st.sidebar.radio('Family Size', ['Greater than 3', '3 or less']) == 'Greater than 3' else 0
user_input['Parental_Status'] = 1 if st.sidebar.radio('Parental Status', ['Together', 'Apart']) == 'Apart' else 0

# Parent Job Information
st.sidebar.subheader("Parent Occupation")
mother_job = st.sidebar.selectbox("Mother's Job", ['health', 'services', 'at_home', 'other', 'teacher'])
father_job = st.sidebar.selectbox("Father's Job", ['health', 'services', 'at_home', 'other', 'teacher'])

# School Information
st.sidebar.subheader("Academic Background")
school_reason = st.sidebar.selectbox("Reason for Choosing School", ['course', 'reputation', 'home', 'other'])
guardian = st.sidebar.selectbox('Guardian', ['mother', 'father', 'other'])

# Feature encoding
jobs = ['health', 'services', 'at_home', 'other', 'teacher']
reasons = ['course', 'reputation', 'home', 'other']
guardians = ['mother', 'father', 'other']

for job in jobs:
    user_input[f'Mother_Job_{job}'] = 1 if mother_job == job else 0
    user_input[f'Father_Job_{job}'] = 1 if father_job == job else 0

for reason in reasons:
    user_input[f'Reason_for_Choosing_School_{reason}'] = 1 if school_reason == reason else 0

for g in guardians:
    user_input[f'Guardian_{g}'] = 1 if guardian == g else 0

# Academic Information
st.sidebar.subheader("Academic Performance")
user_input['Age'] = st.sidebar.slider('Age', 15, 25, 18)
user_input['Study_Time'] = st.sidebar.slider('Study Time (1-4)', 1, 4, 2)
user_input['Failures'] = st.sidebar.slider('Past Failures', 0, 3, 0)
user_input['Absences'] = st.sidebar.slider('Total Absences', 0, 95, 5)

# Grades
user_input['G1'] = st.sidebar.slider('Grade 1 (0-20)', 0, 20, 10)
user_input['G2'] = st.sidebar.slider('Grade 2 (0-20)', 0, 20, 10)
user_input['G3'] = st.sidebar.slider('Grade 3 (0-20)', 0, 20, 10)

# Support Systems
st.sidebar.subheader("Support Systems")
binary_features = {
    'School_Support': 'School Support',
    'Family_Support': 'Family Support',
    'Extra_Paid_Class': 'Extra Classes',
    'Extra_Curricular_Activities': 'Extracurriculars',
    'Internet_Access': 'Internet Access',
    'Attended_Nursery': 'Attended Nursery',
    'Wants_Higher_Education': 'Wants Higher Education',
    'In_Relationship': 'In Relationship'
}

for feature, label in binary_features.items():
    user_input[feature] = 1 if st.sidebar.radio(label, ['Yes', 'No']) == 'Yes' else 0

# Prediction
input_df = pd.DataFrame([user_input]).reindex(columns=expected_features, fill_value=0)

if st.sidebar.button("🚀 Predict Dropout Risk"):
    with st.spinner('Analyzing...'):
        time.sleep(1.5)
        proba = best_model.predict_proba(input_df)[0]
        dropout_prob = round(proba[1]*100, 2)
        stay_prob = round(proba[0]*100, 2)

    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"<div class='metric-box' style='background-color: #e8f5e9;'>"
            f"<h3>🎓 Staying</h3><h2 style='color: #4CAF50;'>{stay_prob}%</h2></div>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"<div class='metric-box' style='background-color: #ffebee;'>"
            f"<h3>🚨 Dropping Out</h3><h2 style='color: #f44336;'>{dropout_prob}%</h2></div>",
            unsafe_allow_html=True
        )
    
    st.progress(stay_prob/100)
    if dropout_prob > 50:
        st.error("⚠️ High Dropout Risk Detected")
    else:
        st.success("🎉 Low Dropout Risk")
    st.markdown("</div>", unsafe_allow_html=True)

# Plot configuration
plt.rcParams['text.color'] = '#2d3436'
plt.rcParams['axes.labelcolor'] = '#2d3436'

