import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path
from model.cleandata import clean_data
from model.createmodel import create_model
from model.qa import answer_query
from streamlit_option_menu import option_menu

# ==============================
# ✅ PATHS (FIXED)
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DATA_PATHS = [
    PROJECT_ROOT / "Data" / "data.csv",
    PROJECT_ROOT / "data.csv",
    PROJECT_ROOT / "DATA" / "Data" / "data.csv",
]
DATA_PATH = next((str(p) for p in CANDIDATE_DATA_PATHS if p.exists()), str(CANDIDATE_DATA_PATHS[0]))
MODEL_PATH = str(PROJECT_ROOT / "model" / "model.pkl")
Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

# ==============================
# ✅ AUTO-TRAIN MODEL
# ==============================
if not os.path.exists(MODEL_PATH):
    if os.path.exists(DATA_PATH):
        df_auto = pd.read_csv(DATA_PATH)
        cleaned_auto = clean_data(df_auto)
        model_auto, _ = create_model(cleaned_auto)
        joblib.dump(model_auto, MODEL_PATH)
        print(f" Model trained automatically from {DATA_PATH}")
    else:
        print(f" Dataset not found at any candidate path: {CANDIDATE_DATA_PATHS}")

# ==============================
# ✅ LOAD MODEL
# ==============================
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

# ==============================
# ✅ LOAD DEFAULT DATA
# ==============================
df = None
cleaned = None

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    cleaned = clean_data(df)

# ==============================
# ✅ STREAMLIT PAGE
# ==============================
st.set_page_config(page_title="AI Model Dashboard", layout="wide")

with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Home", "Train Model", "Insights"],
        icons=["house", "gear", "bar-chart"],
    )

st.title(" AI Model Dashboard")

# ==============================
# ✅ STATUS
# ==============================
if model:
    st.success(" Model ready (auto-trained from data.csv)")
else:
    st.warning(" Model not available")

# ==============================
# ✅ HOME PAGE
# ==============================
if selected == "Home":
    st.markdown("""
        <h1 style='text-align:center;'> Breast Cancer Risk Assessment</h1>
        <p style='text-align:center; font-size:18px;'>
        An AI-powered clinical decision support system
        </p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
    "<h2 style='text-align: center;'>👥 Team Members</h2>",
    unsafe_allow_html=True
)


    team_members = [
        ("Maahir Mohammed", "ADS24B00104Y"),
        ("Abdul Rahman Mahmoud", "ADS24B00129Y"),
        ("Janet Aborvor", "ADS24B00156Y"),
        
    ]

    def member_card(name, id_):
        return f"""
        <div style="
            background-color:#1f2937;
            color:white;
            padding:25px;
            border-radius:20px;
            margin-bottom:20px;
        ">
            <h4 style='margin:0;'>{name}</h4>
            <p style='margin:0;'>ID: {id_}</p>
        </div>
        """

    col1, col2 = st.columns(2)
    for i, member in enumerate(team_members):
        if i % 2 == 0:
            with col1:
                st.markdown(member_card(*member), unsafe_allow_html=True)
        else:
            with col2:
                st.markdown(member_card(*member), unsafe_allow_html=True)

    st.header("Dataset Overview")

    if cleaned is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sample Data")
            st.dataframe(cleaned.head())
        with col2:
            st.subheader("Statistics")
            st.write(cleaned.describe())
    elif os.path.exists(DATA_PATH):
        st.error(f"Dataset found at {DATA_PATH}, but failed to clean it.")
    else:
        st.error(f"Dataset not found at {DATA_PATH}. Upload a CSV on Train Model or place your file there.")

# ==============================
# ✅ TRAIN MODEL PAGE
# ==============================
if selected == "Train Model":
    st.header("⚙️ Train Model from Uploaded CSV")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        st.subheader("Raw Data")
        st.write(data.head())

        cleaned_upload = clean_data(data)

        st.subheader("Cleaned Data")
        st.write(cleaned_upload.head())

        if st.button(" Train Model"):
            model, metrics = create_model(cleaned_upload)

            joblib.dump(model, MODEL_PATH)

            st.success(" Model Trained & Saved")

            st.subheader("Metrics")
            st.write({"Accuracy": metrics.get("accuracy")})
            st.json(metrics.get("classification_report"))
            st.write(metrics.get("confusion_matrix"))

# ==============================
# ✅ INSIGHTS PAGE
# ==============================
if selected == "Insights":
    st.header(" Insights & Predictions")

    tab1, tab2, tab3 = st.tabs(["Predictions", "Ask AI", "Manual Prediction"])

    # ----------------------------
    # ✅ TAB 1: PREDICTIONS
    # ----------------------------
    with tab1:
        pred_file = st.file_uploader("Upload CSV for predictions", type=["csv"])

        if pred_file:
            pred_df = pd.read_csv(pred_file)
            st.write(pred_df.head())

            try:
                pred_clean = clean_data(pred_df)
            except Exception as e:
                st.error(f"Cleaning error: {e}")
                pred_clean = None

            if model and pred_clean is not None:
                X = (
                    pred_clean.drop("diagnosis", axis=1)
                    if "diagnosis" in pred_clean.columns
                    else pred_clean
                )

                preds = model.predict(X)
                pred_clean["prediction"] = preds

                st.success("✅ Predictions Generated")
                st.dataframe(pred_clean.head())

                csv = pred_clean.to_csv(index=False).encode()
                st.download_button(
                    "Download Predictions", csv, "predictions.csv"
                )

    # ----------------------------
    # ✅ TAB 2: ASK AI
    # ----------------------------
    with tab2:
        st.subheader(" Ask Questions About Your Data")

        if cleaned is not None:
            st.markdown(
                """
                 Examples:
                - mean of radius_mean  
                - max of area_mean  
                - unique values of diagnosis  
                - correlation between radius_mean and texture_mean  
                """
            )

            question = st.text_input("Enter your question")

            if st.button("Ask"):
                try:
                    answer = answer_query(cleaned, question)
                    st.success(answer)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.error("No dataset loaded")

    # ----------------------------
    # ✅ TAB 3: MANUAL PREDICTION
    # ----------------------------
    with tab3:
        st.subheader("🔮 Predict Single Input")

        if model and cleaned is not None:
            feature_cols = cleaned.columns[:-1]

            inputs = []
            for col in feature_cols:
                val = st.number_input(f"{col}", value=0.0)
                inputs.append(val)

            if st.button("Predict"):
                try:
                    result = model.predict([inputs])
                    st.success(f"Prediction: {result[0]}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")
        else:
            st.warning("Model or dataset not loaded")