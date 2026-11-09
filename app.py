
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

# Website title
st.title("🚢 Titanic Survival Predictor")

st.write(
    "Enter passenger details to predict the estimated "
    "chance of survival using a Machine Learning model."
)

# Passenger inputs
st.subheader("Passenger Information")

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.radio(
    "Gender",
    ["Female", "Male"],
    horizontal=True
)

age = st.slider(
    "Passenger Age",
    1, 80, 30
)

fare = st.slider(
    "Ticket Fare",
    0.0, 500.0, 30.0
)

# Prediction button
if st.button("Predict Survival", use_container_width=True):

    try:
        model = load_model()

        # Prepare input in the same feature order
        input_data = pd.DataFrame(
            [[pclass, 1 if sex == "Female" else 0, age, fare]],
            columns=["pclass", "sex", "age", "fare"]
        )

        # Predict survival probability
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")

        st.metric(
            "Estimated Survival Probability",
            f"{probability:.1%}"
        )

        if probability >= 0.5:
            st.success("The model predicts: Likely to Survive")
        else:
            st.error("The model predicts: Unlikely to Survive")

    except FileNotFoundError:
        st.error("Model file is missing. Please add model.joblib.")

    except Exception as e:
        st.error(f"Unable to make prediction: {e}")

# Footer
st.divider()

st.caption(
    "Educational Machine Learning Project | "
    "Model trained using the Titanic dataset. "
    "This is a historical-data exercise, not real-world advice."
)
