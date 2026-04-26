import streamlit as st
import importlib
import pandas as pd
import json
import os

# Page config
st.set_page_config(page_title="News Classifier", layout="centered")

st.title("📰 News Document Classifier")

# Input methods
input_method = st.radio(
    " Choose Input Method",
    ["Upload File", "Paste Text"],
    horizontal=True
)
text = ""

if input_method == "Upload File":
    from PyPDF2 import PdfReader

    text = ""

    uploaded_file = st.file_uploader(
        "Upload a file (.txt or .pdf)",
        type=["txt", "pdf"]
    )

    if uploaded_file:
        if uploaded_file.type == "text/plain":
            # TXT file
            text = uploaded_file.read().decode("utf-8")

        elif uploaded_file.type == "application/pdf":
            # PDF file
            pdf_reader = PdfReader(uploaded_file)
            pdf_text = []

            for page in pdf_reader.pages:
                content = page.extract_text()
                if content:
                    pdf_text.append(content)

            text = " ".join(pdf_text)

        st.success("File loaded successfully!")

else:
    text = st.text_area("Paste your news text here")

# Model selection
model_option = st.selectbox(
    "Select Model",
    ["SVM", "Logistic Regression", "Naive Bayes"]
)

# Load model dynamically
def load_model(model_name):
    if model_name == "Logistic Regression":
        return importlib.import_module("models.model_logistic_regression")
    elif model_name == "Naive Bayes":
        return importlib.import_module("models.model_naive_bayes")
    elif model_name == "SVM":
        return importlib.import_module("models.model_linear_svm")

model = load_model(model_option)

# Predict button
if st.button("Predict Category"):
    if text.strip() == "":
        st.warning("Please provide input text")
    else:
        prediction = model.predict(text)
        st.success(f"Predicted Category: **{prediction}**")



if st.button("Compare All Models"):
    if text.strip() == "":
        st.warning("Please provide input text")
    else:
        st.markdown("## Model Comparison")

        model_names = ["SVM", "Logistic Regression", "Naive Bayes"]

        results = []

        for name in model_names:
            try:
                m = load_model(name)
                pred = m.predict(text)
                results.append((name, pred, "N/A"))

            except Exception as e:
                results.append((name, "Error", "-"))

        # Display in columns
        cols = st.columns(3)

        for i, (name, pred, conf) in enumerate(results):
            with cols[i]:
                st.markdown(f"### {name}")
                st.write(f"**Prediction:** {pred}")

        st.markdown("---")

        # ACCURACY COMPARISON
        st.markdown("## Model Accuracy (Test Set)")

        if os.path.exists("model_results.json"):
            with open("model_results.json", "r") as f:
                accuracy_data = json.load(f)

            import pandas as pd
            df = pd.DataFrame(
                list(accuracy_data.items()),
                columns=["Model", "Accuracy"]
            )

            st.table(df)
            st.bar_chart(df.set_index("Model"))

        else:
            st.warning("Run compare_models.py to generate accuracy results")
