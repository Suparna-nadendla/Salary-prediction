import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# Paths
MODEL_PATH = "/Users/suparna/Desktop/best___salary___model.pkl"
DATA_PATH = "eda_data.csv"
USER_DB_PATH = "users.csv"

# Load model and data
model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)

# Page config
st.set_page_config(page_title="Salary Predictor", layout="wide")

# --------------------------- Styling ---------------------------
def set_login_bg():
    st.markdown("""
    <style>
    .stApp {
        background-image: url("https://tse4.mm.bing.net/th?id=OIP.aNntAO59Xpjk6UQLA-YkWwHaEU&pid=Api&P=0&h=180");
        background-size: cover;
        background-attachment: fixed;
    }
    </style>
    """, unsafe_allow_html=True)

def set_main_bg():
    st.markdown("""
    <style>
    .stApp {
        background-image: url("https://tse4.mm.bing.net/th?id=OIP.aNntAO59Xpjk6UQLA-YkWwHaEU&pid=Api&P=0&h=180");
        background-size: cover;
        background-attachment: fixed;
    }
    </style>
    """, unsafe_allow_html=True)

def set_global_style():
    st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        font-size: 18px;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: white !important;
    }
    .stDataFrame, .stTable {
        background-color: rgba(0,0,0,0.5);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------- Authentication ----------------------
def load_users():
    if not os.path.exists(USER_DB_PATH):
        pd.DataFrame(columns=["username", "password"]).to_csv(USER_DB_PATH, index=False)
    return pd.read_csv(USER_DB_PATH)

def login(username, password):
    users = load_users()
    return not users[(users["username"] == username) & (users["password"] == password)].empty

def signup(username, password):
    users = load_users()
    if username in users["username"].values:
        return False
    new_user = pd.DataFrame([[username, password]], columns=["username", "password"])
    new_user.to_csv(USER_DB_PATH, mode='a', header=False, index=False)
    return True

def login_page():
    st.image("https://repository-images.githubusercontent.com/430107293/2170a61e-c3ef-40af-b427-62901ee3d6bd", use_column_width=True)
    st.title("🔐 Login / Signup")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if login(u, p):
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials!")

    with tab2:
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")
        if st.button("Register"):
            if signup(u, p):
                st.success("Registered. Please log in.")
            else:
                st.warning("Username already exists.")

# ---------------------- Pages ----------------------
def show_raw_data():
    st.subheader("📄 Raw Dataset")
    st.dataframe(data)
    st.markdown("### 📊 Statistical Summary")
    st.dataframe(data.describe())

def show_visualizations():
    st.subheader("📊 Data Visualizations")

    col1, col2 = st.columns(2)
    with col1:
        sns.histplot(data['avg_salary'], kde=True, bins=30)
        plt.title("Salary Distribution")
        st.pyplot(plt.gcf())
        plt.clf()

    with col2:
        order = data.groupby('job_simp')['avg_salary'].mean().sort_values(ascending=False).index
        sns.barplot(data=data, y='job_simp', x='avg_salary', order=order)
        plt.title("Average Salary by Job Role")
        st.pyplot(plt.gcf())
        plt.clf()

    st.subheader("📌 Correlation Heatmap")
    corr = data.select_dtypes(include=np.number).corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    st.pyplot(plt.gcf())
    plt.clf()

def show_prediction():
    st.subheader("🧠 Predict Salary")
    with st.form("predict_form"):
        job_title = st.selectbox("Job Title", sorted(data['Job Title'].dropna().unique()))
        location = st.selectbox("Location", sorted(data['Location'].dropna().unique()))
        headquarters = st.selectbox("Headquarters", sorted(data['Headquarters'].dropna().unique()))
        company_name = st.text_input("Company Name", value="ABC Corp")
        company_txt = st.selectbox("Company Type", sorted(data['company_txt'].dropna().unique()))
        founded = st.number_input("Year Founded", value=2000)
        revenue = st.text_input("Revenue", value="Unknown / Non-Applicable")
        min_salary = st.number_input("Minimum Salary", value=60)
        max_salary = st.number_input("Maximum Salary", value=100)
        hourly = st.selectbox("Is Hourly Pay?", [0, 1])
        employer_provided = st.selectbox("Employer Provided Salary?", [0, 1])
        desc_len = st.slider("Job Description Length", 0, 5000, 1000)
        same_state = st.selectbox("Job State same as HQ?", [0, 1])
        rating = st.slider("Company Rating", 0.0, 5.0, 3.5)
        age = st.slider("Company Age", 0, 100, 10)
        python = st.selectbox("Python Skill", [0, 1])
        excel = st.selectbox("Excel Skill", [0, 1])
        aws = st.selectbox("AWS Skill", [0, 1])
        spark = st.selectbox("Spark Skill", [0, 1])
        r_skill = st.selectbox("R Skill", [0, 1])
        num_comp = st.slider("Number of Competitors", 0, 10, 2)

        submitted = st.form_submit_button("Predict")
        if submitted:
            input_dict = {
                'Job Title': job_title,
                'Location': location,
                'Headquarters': headquarters,
                'Company Name': company_name,
                'company_txt': company_txt,
                'Founded': founded,
                'Revenue': revenue,
                'min_salary': min_salary,
                'max_salary': max_salary,
                'hourly': hourly,
                'employer_provided': employer_provided,
                'desc_len': desc_len,
                'same_state': same_state,
                'Rating': rating,
                'age': age,
                'python_yn': python,
                'R_yn': r_skill,
                'spark': spark,
                'aws': aws,
                'excel': excel,
                'num_comp': num_comp,
                'job_state': 'CA',
                'Size': '1001 to 5000 employees',
                'Type of ownership': 'Private',
                'job_simp': 'data scientist',
                'seniority': 'na',
                'Industry': 'Information Technology',
                'Sector': 'Tech'
            }
            input_df = pd.DataFrame([input_dict])
            salary = model.predict(input_df)[0]
            st.success(f"💰 Predicted Salary: ₹{salary:.2f}K")

def show_batch_prediction():
    st.subheader("📁 Batch Prediction from CSV")
    uploaded_file = st.file_uploader("Upload CSV with required columns", type=['csv'])
    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file)
            predictions = model.predict(df_upload)
            df_upload['Predicted Salary'] = predictions
            st.dataframe(df_upload)
            csv = df_upload.to_csv(index=False).encode()
            st.download_button("📥 Download Predictions", csv, "predicted_salaries.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

# ------------------ App Flow ------------------
set_global_style()
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    set_login_bg()
    login_page()
else:
    set_main_bg()
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()
    nav_option = st.sidebar.radio("Navigation", ["📄 Data", "📊 Visuals", "🧠 Predict", "📁 CSV Upload"])
    if nav_option == "📄 Data":
        show_raw_data()
    elif nav_option == "📊 Visuals":
        show_visualizations()
    elif nav_option == "🧠 Predict":
        show_prediction()
    elif nav_option == "📁 CSV Upload":
        show_batch_prediction()
