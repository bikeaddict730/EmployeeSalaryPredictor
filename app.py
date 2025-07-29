import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load data and trained model
data = pd.read_csv('Salary Data.csv')
model = pickle.load(open('salary_model.pkl', 'rb'))

# Prepare encoded columns for Job Title and Gender
job_title_dummies = pd.get_dummies(data['Job Title'], drop_first=True)
gender_dummies = pd.get_dummies(data['Gender'], drop_first=True)

job_title_columns = job_title_dummies.columns
gender_columns = gender_dummies.columns

# Streamlit page config
st.set_page_config(page_title="Employee Salary Predictor", layout="wide")
st.title("💼 Employee Salary Predictor")

# Sidebar inputs for prediction
st.sidebar.header("Enter Employee Details")
age = st.sidebar.number_input("Age", min_value=18, max_value=65, value=30)
experience = st.sidebar.number_input("Years of Experience", min_value=0, max_value=40, value=5)
job_title = st.sidebar.selectbox("Job Title", data['Job Title'].unique())
gender = st.sidebar.selectbox("Gender", data['Gender'].unique())

# Predict salary button
if st.sidebar.button("Predict Salary"):
    input_dict = {
        'Age': [age],
        'Years of Experience': [experience]
    }

    # Encode job title dummies
    for col in job_title_columns:
        input_dict[col] = [1 if col == job_title else 0]

    # Encode gender dummies
    for col in gender_columns:
        input_dict[col] = [1 if col == gender else 0]

    input_df = pd.DataFrame(input_dict)

    # Ensure all model input columns are present
    missing_cols = set(model.feature_names_in_) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[model.feature_names_in_]

    # Predict salary
    salary = model.predict(input_df)[0]

    # Display result with size and animation
    st.markdown(f"""
    <div style="background-color:#dff0d8; padding: 20px; border-radius: 10px; text-align: center; animation: fadeInScale 1s ease-in-out;">
        <h2 style="color: #3c763d; font-size: 36px; margin: 0;">💰 Predicted Salary:</h2>
        <p style="color: #3c763d; font-size: 48px; font-weight: bold; margin: 10px 0;">${salary:,.2f}</p>
    </div>

    <style>
    @keyframes fadeInScale {{
      0% {{
        opacity: 0;
        transform: scale(0.8);
      }}
      100% {{
        opacity: 1;
        transform: scale(1);
      }}
    }}
    </style>
    """, unsafe_allow_html=True)

# --- Data Visualisation Section ---

st.markdown("---")
st.header("📊 Data Visualisation")

# Initialize toggle states for graphs
graph_states = {
    'graph1': 'show_salary_age',
    'graph2': 'show_salary_experience',
    'graph3': 'show_jobtitle_count',
    'graph4': 'show_salary_dist'
}

for key in graph_states.values():
    if key not in st.session_state:
        st.session_state[key] = False

# Layout for graphs
col1, col2 = st.columns(2)

# Graph 1: Salary vs Age
with col1:
    if st.button("📈 Toggle Salary vs Age"):
        st.session_state.show_salary_age = not st.session_state.show_salary_age
    if st.session_state.show_salary_age:
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x='Age', y='Salary', hue='Gender', ax=ax)
        plt.title("Salary vs Age")
        st.pyplot(fig)

# Graph 2: Salary vs Years of Experience
with col2:
    if st.button("📈 Toggle Salary vs Years of Experience"):
        st.session_state.show_salary_experience = not st.session_state.show_salary_experience
    if st.session_state.show_salary_experience:
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x='Years of Experience', y='Salary', hue='Gender', ax=ax)
        plt.title("Salary vs Years of Experience")
        st.pyplot(fig)

# Graph 3: Count of Employees by Job Title
with col1:
    if st.button("📊 Toggle Count of Employees by Job Title"):
        st.session_state.show_jobtitle_count = not st.session_state.show_jobtitle_count
    if st.session_state.show_jobtitle_count:
        fig, ax = plt.subplots()
        sns.countplot(data=data, x='Job Title', order=data['Job Title'].value_counts().index, ax=ax)
        plt.xticks(rotation=45)
        plt.title("Employee Count by Job Title")
        st.pyplot(fig)

# Graph 4: Salary Distribution (Histogram)
with col2:
    if st.button("📊 Toggle Salary Distribution"):
        st.session_state.show_salary_dist = not st.session_state.show_salary_dist
    if st.session_state.show_salary_dist:
        fig, ax = plt.subplots()
        sns.histplot(data['Salary'], kde=True, ax=ax)
        plt.title("Salary Distribution")
        st.pyplot(fig)
