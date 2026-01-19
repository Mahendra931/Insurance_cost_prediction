# import streamlit as st
# import joblib
# import numpy as np

# # Load trained model
# # model = joblib.load("model/linear_regression.pkl")

# model = joblib.load("model/random_forest.pkl")


# st.title("Insurance Cost Prediction")
# st.write("Predict insurance charges using Linear Regression")

# # User inputs
# age = st.number_input("Age", min_value=18, max_value=100)
# sex = st.selectbox("Sex", ["Male", "Female"])
# bmi = st.number_input("BMI", min_value=10.0, max_value=50.0)
# children = st.number_input("Number of Children", min_value=0, max_value=5)
# smoker = st.selectbox("Smoker", ["Yes", "No"])
# region = st.selectbox(
#     "Region",
#     ["northwest", "southeast", "southwest", "northeast"]
# )

# # Encode inputs (same logic as training)
# sex = 1 if sex == "Female" else 0
# smoker = 1 if smoker == "Yes" else 0

# # Region dummies (drop_first=True logic)
# region_northwest = 1 if region == "northwest" else 0
# region_southeast = 1 if region == "southeast" else 0
# region_southwest = 1 if region == "southwest" else 0
# # northeast is the dropped category → all zeros

# # Prediction
# if st.button("Predict Insurance Cost"):
#     input_data = np.array([[
#         age,
#         sex,
#         bmi,
#         children,
#         smoker,
#         region_northwest,
#         region_southeast,
#         region_southwest
#     ]])

#     prediction = model.predict(input_data)
#     st.success(f"Predicted Insurance Cost: ${prediction[0]:.2f}")






import streamlit as st
import joblib
import numpy as np

# ✅ MUST be the first Streamlit command
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="💰",
    layout="centered"
)



# Custom CSS (AFTER set_page_config)
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            to right,
            #0f2027,
            #203a43,
            #2c5364
        );
        color: white;
    }

    label {
        color: #f0f0f0 !important;
        font-weight: 500;
    }

    div.stButton > button {
        background-color: #1db954;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-size: 16px;
        font-weight: bold;
    }

    div.stButton > button:hover {
        background-color: #17a74a;
        color: white;
    }

    .stAlert {
        border-radius: 10px;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
model = joblib.load("model/random_forest.pkl")


# App content
st.title("🏥 SecureLife Insurance")
st.markdown(
    "Instantly estimate your health insurance premium using our smart pricing system."
)

st.divider()

# Inputs
st.subheader("📋 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100)
    bmi = st.number_input("BMI", 10.0, 50.0)
    children = st.number_input("Number of Children", 0, 5)

with col2:
    sex = st.selectbox("Sex", ["Male", "Female"])
    smoker = st.selectbox("Smoker", ["Yes", "No"])
    region = st.selectbox(
        "Region",
        ["northeast", "northwest", "southeast", "southwest"]
    )

# Encoding
sex = 1 if sex == "Female" else 0
smoker = 1 if smoker == "Yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

st.divider()

# Prediction
if st.button("🔍 Predict Insurance Cost", use_container_width=True):
    input_data = np.array([[
        age,
        sex,
        bmi,
        children,
        smoker,
        region_northwest,
        region_southeast,
        region_southwest
    ]])

    prediction = model.predict(input_data)[0]

    st.success(f"💵 Estimated Insurance Cost: **${prediction:,.2f}**")
    st.caption(
        "ℹ️ This is an estimate. Actual charges may vary based on provider policies."
    )


import pandas as pd
feature_names = [
    "age", "sex", "bmi", "children", "smoker",
    "region_northwest", "region_southeast", "region_southwest"
]

importances = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

st.subheader("📊 Feature Importance")
st.bar_chart(importance_df.set_index("Feature"))
