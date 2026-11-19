from pathlib import Path
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_FILES = {
    "Naive Bayes": "naive_bayes.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "SVM": "svm.joblib",
    "Random Forest": "random_forest.joblib",
}

st.set_page_config(page_title="Spam Email Detector", page_icon="📧")
st.title("📧 Spam Email Detection")
st.write("Enter an email message and choose a trained machine-learning model.")

available = {
    name: path for name, filename in MODEL_FILES.items()
    if (path := MODEL_DIR / filename).exists()
}
if not available:
    st.warning("No trained models found. Run `python train.py` first.")
    st.stop()

model_name = st.selectbox("Choose model", list(available))
email_text = st.text_area("Email subject and body", height=180,
                          placeholder="Paste the email text here...")
if st.button("Check email"):
    if not email_text.strip():
        st.error("Please enter some email text.")
    else:
        model = joblib.load(available[model_name])
        prediction = model.predict([email_text])[0]
        if prediction == "spam":
            st.error("Prediction: SPAM")
        else:
            st.success("Prediction: HAM (Not Spam)")
        st.caption(f"Model used: {model_name}. This is a prediction, not a guarantee.")
