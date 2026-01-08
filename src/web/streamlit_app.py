import streamlit as st
import torch
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.model import AutoJudge
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model

# --- CONFIGURATION ---
# Path to your saved model weights (check both locations)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH_ROOT = os.path.join(PROJECT_ROOT, "autojudge_best_acc_model.pth")
MODEL_PATH_MODELS = os.path.join(PROJECT_ROOT, "model_weights", "autojudge_best_acc_model.pth")
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LEN = 350
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CACHED MODEL LOADING ---
@st.cache_resource
def load_model():
    # 1. Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        return None, None

    # 2. Initialize Model Structure
    model = AutoJudge(MODEL_NAME)
    
    # 3. Apply LoRA Config (Must match training config)
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["query_proj", "key_proj", "value_proj", "dense"], 
        lora_dropout=0.1,
        bias="none"
    )
    model.bert = get_peft_model(model.bert, peft_config)
    
    # 4. Load Weights (check both root and models/ folder)
    model_path = None
    if os.path.exists(MODEL_PATH_ROOT):
        model_path = MODEL_PATH_ROOT
    elif os.path.exists(MODEL_PATH_MODELS):
        model_path = MODEL_PATH_MODELS
    else:
        st.error(f"❌ Model file not found at either:")
        st.error(f"   - {MODEL_PATH_ROOT}")
        st.error(f"   - {MODEL_PATH_MODELS}")
        st.info("Please ensure 'autojudge_best_acc_model.pth' exists in the project root or 'models/' folder.")
        return None, None
    
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.info(f"Attempted to load from: {model_path}")
        return None, None
    
    model.to(DEVICE)
    model.eval()
    
    return tokenizer, model

# --- UI LAYOUT ---
st.set_page_config(page_title="AutoJudge", page_icon="⚖️", layout="wide")

# Header
st.title("⚖️ AutoJudge: AI Task Complexity Evaluator")
st.markdown("""
**Automated Complexity Analysis for Programming Problems** *Powered by DeBERTa-v3-Large + LoRA*
""")

st.divider()

# Layout: Two Columns (Left for Input, Right for Output)
col_input, col_output = st.columns([1, 1])

with col_input:
    st.subheader("📝 Problem Details")
    
    title = st.text_input("Task Title", placeholder="e.g., Shortest Path in a Grid")
    
    desc = st.text_area(
        "Problem Description", 
        height=200, 
        placeholder="Paste the main problem statement here..."
    )
    
    with st.expander("Additional Details (Optional)"):
        input_desc = st.text_area("Input Constraints", placeholder="e.g., N <= 10^5...")
        output_desc = st.text_area("Output Format", placeholder="e.g., Print a single integer...")

    predict_btn = st.button("🚀 Analyze Complexity", type="primary", use_container_width=True)

# Processing
if predict_btn:
    if not desc.strip():
        st.warning("⚠️ Please enter a Problem Description to proceed.")
    else:
        with col_output:
            st.subheader("📊 Analysis Results")
            
            with st.spinner("🧠 Loading Model & Analyzing..."):
                tokenizer, model = load_model()
                
                if model:
                    # Construct Input Text (must match training format from data_loader.py)
                    # Format: title + " [SEP] " + description + " [SEP] " + input_description + " " + output_description
                    full_text = f"{title or ''} [SEP] {desc or ''} [SEP] {input_desc or ''} {output_desc or ''}".strip()
                    
                    # Tokenize
                    enc = tokenizer(
                        full_text,
                        max_length=MAX_LEN,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )
                    
                    ids = enc['input_ids'].to(DEVICE)
                    mask = enc['attention_mask'].to(DEVICE)
                    
                    # Inference
                    with torch.no_grad():
                        logits_cls, score = model(ids, mask)
                        
                        # Process Class Prediction
                        pred_class_idx = torch.argmax(logits_cls, dim=1).item()
                        class_map = {0: "Easy", 1: "Medium", 2: "Hard"}
                        predicted_class = class_map.get(pred_class_idx, "Unknown")
                        
                        # Process Score Prediction (handle dimension: score shape is [batch, 1])
                        pred_score = score.squeeze().item()

                    # --- Display Results ---
                    
                    # 1. Difficulty Card
                    if predicted_class == "Easy":
                        st.success(f"### Difficulty: {predicted_class}")
                    elif predicted_class == "Medium":
                        st.warning(f"### Difficulty: {predicted_class}")
                    else:
                        st.error(f"### Difficulty: {predicted_class}")
                        
                    # 2. Complexity Score
                    st.metric(
                        label="Computational Complexity Score (1.0 - 10.0)", 
                        value=f"{pred_score:.2f}"
                    )
                    
                    # Visual Progress Bar for Score
                    # Normalize score 1-10 to 0-1 for progress bar
                    norm_score = max(0.0, min(1.0, (pred_score - 1.0) / 9.0))
                    st.progress(norm_score)
                    
                    st.divider()
                    
                    # 3. Technical Debug Info
                    with st.expander("🔍 Technical Debug Info"):
                        st.json({
                            "Logits (Easy, Medium, Hard)": logits_cls.cpu().numpy().tolist(),
                            "Raw Regression Output": pred_score,
                            "Input Token Count": torch.sum(mask).item()
                        })

# Footer
st.markdown("---")
st.caption("AutoJudge Project | Enrollment No: 24117002")

