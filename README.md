# AI Model Dashboard

Quick steps to run the Streamlit app locally:

1. Create and activate a Python environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. From the project root run:

```bash
python -m streamlit run model/app.py
```

4. In the app: upload your CSV, go to "Train Model", click "Train Model" to train and save the model. Use "Insights" to upload CSVs and get predictions.

Model is saved to `model/model.pkl` after training.