
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ----------- LOGIN SYSTEM ------------
users = {
    "moheed": "ali",
    "vansh": "sharma",
    "arthav": "gupta",
    "ayon": "das",
    "somil": "dadwal"
}

def login():
    attempts = 3
    while attempts > 0:
        username = input("Enter username: ").strip().lower()
        password = input("Enter password: ").strip().lower()
        if username in users and users[username] == password:
            print(f"\n Welcome, {username}!\n")
            return True
        else:
            attempts -= 1
            print(f" Invalid credentials. {attempts} attempts remaining.\n")
    print(" Too many failed attempts. Exiting program.")
    return False

if not login():
    exit()

# -------------------- DATA LOADING --------------------
df = pd.read_csv("StudentsPerformance.csv")

print(" First 5 Records:\n")
print(df.head())

print("\n Dataset Info:\n")
print(df.info())

print("\n Statistical Summary:\n")
print(df.describe())

# -------------------- VISUALIZATIONS --------------------
plt.figure(figsize=(6, 5))
sns.boxplot(x='gender', y='math score', data=df, palette='Set2')
plt.title('Gender-wise Math Score Distribution')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.scatterplot(x='reading score', y='writing score', data=df, hue='gender')
plt.title('Reading vs. Writing Score')
plt.xlabel('Reading Score')
plt.ylabel('Writing Score')
plt.tight_layout()
plt.show()

# -------------------- DATA ENCODING --------------------
df_encoded = df.copy()
label_encoder = LabelEncoder()

for column in df_encoded.select_dtypes(include='object'):
    df_encoded[column] = label_encoder.fit_transform(df_encoded[column])

# -------------------- TRAINING MODEL --------------------
X = df_encoded.drop(columns=['math score'])
y = df_encoded['math score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# -------------------- EVALUATION --------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\n Model Performance:")
print(f"R² Score: {r2:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

# -------------------- PREDICTIONS --------------------
compare_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

print("\n Sample Predictions:\n")
print(compare_df.head(10))

# -------------------- PLOTS --------------------
plt.figure(figsize=(6, 5))
sns.scatterplot(x='Actual', y='Predicted', data=compare_df, alpha=0.7)
plt.plot([0, 100], [0, 100], '--', color='red')
plt.title('Actual vs Predicted Math Score')
plt.xlabel('Actual Math Score')
plt.ylabel('Predicted Math Score')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(compare_df['Actual'][:25].values, label='Actual', marker='o')
plt.plot(compare_df['Predicted'][:25].values, label='Predicted', marker='x')
plt.title('Actual vs Predicted (First 25 Samples)')
plt.xlabel('Sample Index')
plt.ylabel('Math Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
compare_df[:10].plot(kind='bar')
plt.title('Bar Chart: Actual vs Predicted (First 10 Samples)')
plt.ylabel('Math Score')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
