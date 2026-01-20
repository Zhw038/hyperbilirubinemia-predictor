import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Load the new model
model = joblib.load('ET-20.pkl')

# Load the test data from X_test.csv to create LIME explainer
X_test = pd.read_csv('X_test-20.csv')


feature_names = [
    "weight", "height", "PVD", "PH", "Preoperative.intubation", "TBIL.pre", "Hb", "PLT", "INR", "pre.UA","Aortic.occlusion.time", 
    "surgery.time", "anes.time", "Sevoflurane", "Dexmedetomidine", "Blood.loss", "Urine", "RBC.t", "PLT.t", "Alb.t"
]

# Streamlit user interface
st.title("Hyperbilirubinemia after on-pump Cardiac Surgery Predictor")

# weight: numerical input
weight = st.number_input("weight:", min_value=0.0, max_value=500, value=66.6)

# height: numerical input
height = st.number_input("height:", min_value=0, max_value=300, value=155)

# PVD: categorical selection
pvd = st.selectbox("PVD:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "NO")

# PH: categorical selection
ph = st.selectbox("PH:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "NO")

# Preoperative.intubation: categorical selection
pi = st.selectbox("Preoperative.intubation:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "NO")

# TBIL.pre: numerical input
tb = st.number_input("TBIL.pre:", min_value=0.0, max_value=100.0, value=17.2)

# Hb: numerical input
hb = st.number_input("Hb:", min_value=0, max_value=300, value=138)

# PLT: numerical input
plt = st.number_input("PLT:", min_value=0, max_value=1000, value=188)

# INR: numerical input
inr = st.number_input("INR:", min_value=0.00, max_value=10.00, value=1.72)

# pre.UA: numerical input
ua = st.number_input("pre.UA:", min_value=0, max_value=3000, value=488)

# Aortic.occlusion.time: numerical input
acct = st.number_input("Aortic.occlusion.time:", min_value=0, max_value=1000, value=88)

# surgery.time: numerical input
st = st.number_input("surgery.time:", min_value=0, max_value=3000, value=288)

# anes.time: numerical input
at = st.number_input("anes.time:", min_value=0, max_value=3000, value=388)

# Sevoflurane: categorical selection
sevo = st.selectbox("Sevoflurane:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "NO")

# Dexmedetomidine: categorical selection
dex = st.selectbox("Dexmedetomidine:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "NO")

# Blood.loss: numerical input
bl = st.number_input("Blood.loss:", min_value=0, max_value=10000, value=800)

# Urine: numerical input
urine = st.number_input("Urine:", min_value=0, max_value=10000, value=1000)

# RBC.t: numerical input
rbct = st.number_input("RBC.t:", min_value=0, max_value=10000, value=600)

# PLT.t: numerical input
pltt = st.number_input("PLT.t:", min_value=0.0, max_value=10.0, value=1)

# Alb.t: categorical selection
albt = st.number_input("Alb.t:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "NO")




feature_values = [weight,height,pvd,ph,pi,tb,hb,plt,inr,ua,acct,st,at,sevo,dex,bl,urine,rbct,pltt,albt]
features = np.array([feature_values])


if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class} (1: Disease, 0: No Disease)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of hyperbilirubinemia. "
            f"The model predicts that your probability of having hyperbilirubinemia is {probability:.1f}%. "
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of hyperbilirubinemia. "
            f"The model predicts that your probability of not having hyperbilirubinemia is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
        )

    st.write(advice)

    # SHAP Explanation
    st.subheader("SHAP Force Plot Explanation")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    
    # Display the SHAP force plot for the predicted class
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    else:
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')

#     # LIME Explanation
#     st.subheader("LIME Explanation")
#     lime_explainer = LimeTabularExplainer(
#         training_data=X_test.values,
#         feature_names=X_test.columns.tolist(),
#         class_names=['Not sick', 'Sick'],  # Adjust class names to match your classification task
#         mode='classification'
#     )
    
#     # Explain the instance
#     lime_exp = lime_explainer.explain_instance(
#         data_row=features.flatten(),
#         predict_fn=model.predict_proba
#     )

#     # Display the LIME explanation without the feature value table
#     lime_html = lime_exp.as_html(show_table=False)  # Disable feature value table
#     st.components.v1.html(lime_html, height=800, scrolling=True)
