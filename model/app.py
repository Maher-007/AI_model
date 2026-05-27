import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from model.cleandata import clean_data
from model.createmodel import create_model
from model.qa import answer_query
from streamlit_option_menu import option_menu

st.set_page_config(page_title="AI Model Dashboard", layout="wide")

with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Home", "Train Model", "Insights"],
        icons=["house", "gear", "bar-chart"]
    )

st.title("AI Model Dashboard")

uploaded_file = None
if selected in ("Home", "Train Model"):
    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])    

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.write(data.head())

    # Clean data
    cleaned = clean_data(data)
    st.subheader("Cleaned Data")
    st.write(cleaned.head())

    if selected == "Train Model":
        if st.button("Train Model"):
            model, metrics = create_model(cleaned)

            # save model to disk
            model_path = "model/model.pkl"
            joblib.dump(model, model_path)

            st.success("✅ Model trained and saved to model/model.pkl")

            # show metrics
            st.subheader("Training Metrics")
            st.write({"Accuracy": metrics.get('accuracy')})
            st.write("Classification Report:")
            st.json(metrics.get('classification_report'))
            st.write("Confusion Matrix:")
            st.write(metrics.get('confusion_matrix'))

            # download button for model
            with open(model_path, "rb") as f:
                btn = st.download_button(
                    label="Download trained model",
                    data=f,
                    file_name="model.pkl",
                    mime="application/octet-stream"
                )

    # Show stats
    st.subheader("Statistics")
    st.write(cleaned.describe())

if selected == "Insights":
    st.header("Insights & Predictions")
    st.markdown("Upload a CSV to run predictions using the saved model (`model/model.pkl`).")
    pred_file = st.file_uploader("Upload CSV for predictions", type=["csv"], key="pred")
    if pred_file is not None:
        if not st.sidebar:
            pass
        pred_df = pd.read_csv(pred_file)
        st.subheader("Raw Prediction Data")
        st.write(pred_df.head())

        try:
            pred_clean = clean_data(pred_df)
        except Exception as e:
            st.error(f"Error cleaning prediction data: {e}")
            pred_clean = None

        import os
        model_path = "model/model.pkl"
        if not os.path.exists(model_path):
            st.error("No trained model found at model/model.pkl — train a model first.")
        else:
            model = joblib.load(model_path)
            if pred_clean is not None:
                X = pred_clean.drop('diagnosis', axis=1) if 'diagnosis' in pred_clean.columns else pred_clean
                preds = model.predict(X)
                pred_clean['prediction'] = preds
                st.subheader("Predictions")
                st.write(pred_clean.head())

                # allow downloading predictions
                to_download = pred_clean.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions CSV", to_download, file_name="predictions.csv", mime="text/csv")

        # QA on prediction data or uploaded data
        st.subheader("Ask questions about this dataset")
        question = st.text_input("Enter a question (e.g. 'mean of radius_mean')")
        if st.button("Ask"):
            try:
                source_df = pred_clean if 'pred_clean' in locals() and pred_clean is not None else pred_df
                ans = answer_query(source_df, question)
                st.write(ans)
            except Exception as e:
                st.error(f"Error answering question: {e}")

    else:
        st.info("Upload a CSV in Insights to run predictions or ask questions about the data.")