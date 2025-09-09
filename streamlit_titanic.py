import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import sys
import threading
import time
import os
from sklearn.linear_model import LogisticRegression
from pyngrok import ngrok
from pyngrok.exception import PyngrokNgrokError

# Load the data
@st.cache_data
def load_data():
    """Loads and preprocesses the Titanic dataset."""
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Encode categorical features
    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)

    return df

# Train the model
def train_model(df):
    """Trains a logistic regression model on the data."""
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    y = df['Survived']
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    return model

# Main app logic
def main():
    """Defines the Streamlit app layout and logic."""
    st.set_page_config(
        page_title="Titanic Survival Predictor",
        page_icon="🚢",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Load and train data
    data = load_data()
    model = train_model(data)

    st.title("Titanic Survival Predictor 🚢")
    st.write("Adjust the features below to predict whether a passenger would have survived the Titanic disaster.")
    st.markdown("---")

    # Sidebar for user input
    st.sidebar.header("Passenger Details")
    pclass_options = sorted(data['Pclass'].unique().tolist())
    pclass = st.sidebar.selectbox("Passenger Class", pclass_options)

    sex_options = {'male': 1, 'female': 0}
    sex = st.sidebar.selectbox("Sex", list(sex_options.keys()))

    age = st.sidebar.slider("Age", 0.0, 80.0, float(data['Age'].median()))

    sibsp_options = sorted(data['SibSp'].unique().tolist())
    sibsp = st.sidebar.selectbox("Number of Sibling/Spouses Aboard", sibsp_options)

    parch_options = sorted(data['Parch'].unique().tolist())
    parch = st.sidebar.selectbox("Number of Parents/Children Aboard", parch_options)

    fare = st.sidebar.slider("Fare ($)", 0.0, float(data['Fare'].max()), float(data['Fare'].median()))

    # Create a prediction based on user input
    input_df = pd.DataFrame([[pclass, sex_options[sex], age, sibsp, parch, fare]],
                             columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])

    # Display the user input
    st.subheader("Your Input")
    st.dataframe(input_df.style.set_properties(**{'font-family': 'monospace'}), hide_index=True)

    # Make and display the prediction
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction")

    if prediction == 1:
        st.success("The model predicts that this passenger would have survived! ✅")
        st.balloons()
    else:
        st.error("The model predicts that this passenger would NOT have survived. ❌")

    st.markdown("---")
    st.markdown("### Model Insights")
    st.write("This simple logistic regression model gives us a glimpse into which factors were most important.")

    coefficients = pd.DataFrame(model.coef_.flatten(), index=input_df.columns, columns=['Coefficient']).sort_values('Coefficient', ascending=False)
    st.table(coefficients)

    st.caption("A positive coefficient increases the chance of survival, while a negative one decreases it.")

# Start the Streamlit app and ngrok tunnel
if __name__ == "__main__":
    # The runner logic for the Streamlit app
    def run_app():
        subprocess.run([sys.executable, "-m", "streamlit", "run", sys.argv[0]])

    # Start streamlit in a separate thread
    threading.Thread(target=run_app, daemon=True).start()

    # Use ngrok to create a public URL for the app
    time.sleep(5)  # Give the app some time to start

    # Authenticate ngrok using an environment variable
    ngrok_auth_token = os.getenv('2v8nqm1AhbVyo5bSvCMYTGgkIXy_32tGHakywYSGs436dkpVH')
    if not ngrok_auth_token:
        print("Error: The NGROK_AUTH_TOKEN environment variable is not set.")
        print("Please get your token from https://dashboard.ngrok.com/get-started/your-authtoken and set it as an environment variable.")
        sys.exit(1)
        
    ngrok.set_auth_token(ngrok_auth_token)

    # Kill all existing ngrok tunnels to free up the session limit
    try:
        ngrok.kill()
        public_url = ngrok.connect(8501)
        print(f"\nYour Streamlit app is running at: {public_url}")
    except PyngrokNgrokError as e:
        print("It looks like your ngrok session is already in use. To fix this, please follow these steps:")
        print("1. Go to the menu at the top of the screen and click 'Runtime'.")
        print("2. Select 'Restart session'.")
        print("3. Rerun the code cell once the session has restarted.")
        print(f"Error details: {e}")
