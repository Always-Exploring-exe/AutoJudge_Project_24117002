# Streamlit Local Hosting & Demo Video Guide

## 🚀 Running Streamlit App Locally

### Prerequisites Check

1. **Ensure you have the trained model**:
   ```bash
   # Check if model exists
   ls model_weights/autojudge_best_acc_model.pth
   ```
   If not, train the model first using the notebook in `other_models_regressor_and_classifier_train_and_run` or `regressor_classifier_train_and_run.ipynb` for the best model.

2. **Install dependencies** (if not already done):
   ```bash
   pip install -r requirements.txt
   ```

### Step-by-Step: Launch the App

1. **Open Terminal/Command Prompt** in the project root directory:
   ```bash
   cd C:\Users\Abhi\Downloads\AutoJudge_Project_24117002
   ```

2. **Run Streamlit**:
   ```bash
   streamlit run src/web/streamlit_app.py
   ```

3. **Access the App**:
   - Streamlit will automatically open your browser at `http://localhost:8501`
   - If it doesn't open automatically, copy the URL from the terminal and paste it in your browser

4. **First Run Notes**:
   - First time loading will download the DeBERTa-v3-Large tokenizer (may take a minute)
   - Model loading will take 10-30 seconds depending on your hardware
   - The app uses `@st.cache_resource` so the model loads once and stays in memory

### Troubleshooting

**Issue: Model file not found**
- Make sure `autojudge_best_acc_model.pth` is in the `model_weights/` directory
- Check the terminal output for exact path it's looking for

**Issue: Import errors**
- Make sure you're running from the project root directory
- Try: `python -m streamlit run src/web/streamlit_app.py`

**Issue: Out of memory**
- Close other applications
- The app will use CPU if GPU is unavailable (slower but works)

**Issue: Port already in use**
- Streamlit will suggest an alternative port (e.g., 8502)
- Or kill the process using port 8501

---



