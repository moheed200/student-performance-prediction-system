import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --------------- LOGIN SYSTEM ---------------
users = {
    "moheed": "ali",
    "vansh": "sharma",
    "arthav": "gupta",
    "ayon": "das",
    "somil": "dadwal"
}

def login():
    st.sidebar.title(" Login Panel")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username.lower() in users and users[username.lower()] == password.lower():
            st.session_state["logged_in"] = True
            st.session_state["user"] = username
            st.sidebar.success(f"Welcome, {username}!")
        else:
            st.sidebar.error("Invalid username or password.")

# Check login state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# --------------- MAIN DASHBOARD ---------------
st.title(" Student Performance Prediction Dashboard")
st.write(f"Welcome **{st.session_state['user'].capitalize()}!** 👋")

# Load CSV file
df = pd.read_csv("StudentsPerformance.csv")

st.header(" Dataset Overview")
st.write("Here are the first few rows of the dataset:")
st.dataframe(df.head())

# Dataset info
st.subheader("Dataset Summary")
st.write(df.describe())

# Visualizations
st.header(" Data Visualizations")

# Gender-wise Math Score
st.subheader("Gender-wise Math Score Distribution")
fig1, ax1 = plt.subplots()
sns.boxplot(x='gender', y='math score', data=df, palette='Set2', ax=ax1)
st.pyplot(fig1)

# Reading vs Writing Score
st.subheader("Reading vs Writing Score (Gender-wise)")
fig2, ax2 = plt.subplots()
sns.scatterplot(x='reading score', y='writing score', data=df, hue='gender', ax=ax2)
st.pyplot(fig2)

# Data Encoding
df_encoded = df.copy()
label_encoder = LabelEncoder()
for column in df_encoded.select_dtypes(include='object'):
    df_encoded[column] = label_encoder.fit_transform(df_encoded[column])

# Train-Test Split
X = df_encoded.drop(columns=['math score'])
y = df_encoded['math score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

st.header(" Model Performance")
st.write(f"**R² Score:** {r2:.2f}")
st.write(f"**Mean Squared Error:** {mse:.2f}")

# Compare Actual vs Predicted
compare_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
st.subheader("Actual vs Predicted (First 10 Samples)")
st.dataframe(compare_df.head(10))

# Scatter Plot
st.subheader("Actual vs Predicted Scatter Plot")
fig3, ax3 = plt.subplots()
sns.scatterplot(x='Actual', y='Predicted', data=compare_df, ax=ax3, alpha=0.7)
plt.plot([0, 100], [0, 100], '--', color='red')
st.pyplot(fig3)

# Line Chart
st.subheader("Actual vs Predicted (First 25 Samples)")
fig4, ax4 = plt.subplots(figsize=(10, 4))
ax4.plot(compare_df['Actual'][:25].values, label='Actual', marker='o')
ax4.plot(compare_df['Predicted'][:25].values, label='Predicted', marker='x')
ax4.legend()
ax4.set_xlabel("Sample Index")
ax4.set_ylabel("Math Score")
ax4.grid(True)
st.pyplot(fig4)

# Bar Chart
st.subheader("Bar Chart: Actual vs Predicted (First 10 Samples)")
fig5, ax5 = plt.subplots()
compare_df[:10].plot(kind='bar', ax=ax5)
plt.ylabel("Math Score")
st.pyplot(fig5)

st.success(" Model Training & Visualization Complete!")
