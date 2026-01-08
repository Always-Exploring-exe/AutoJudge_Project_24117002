import requests
import pandas as pd
import os
import io

# Dataset URL
DATASET_URL = "https://raw.githubusercontent.com/AREEG94FAHAD/TaskComplexityEval-24/main/problems_data.jsonl"
# Updated path to point to data directory
LOCAL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "problems_data.jsonl")

def load_data():
    """
    Downloads and prepares the TaskComplexity dataset.
    Returns: cleaned DataFrame.
    """
    # 1. Download
    if not os.path.exists(LOCAL_FILE):
        print(f"📥 Downloading dataset from GitHub...")
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(LOCAL_FILE), exist_ok=True)
        try:
            r = requests.get(DATASET_URL)
            if r.status_code == 200:
                with open(LOCAL_FILE, "wb") as f:
                    f.write(r.content)
                print("✅ Download complete.")
            else:
                print("⚠️ Download failed. Please provide 'problems_data.jsonl' manually.")
                return None
        except Exception as e:
            print(f"❌ Error downloading: {e}")
            return None

    # 2. Parse JSONL
    try:
        df = pd.read_json(LOCAL_FILE, lines=True)
        
        # 3. Feature Engineering: Construct Full Context
        # We concatenate all text fields to give BERT maximum context
        df['full_text'] = (
            df['title'].fillna('') + " [SEP] " + 
            df['description'].fillna('') + " [SEP] " + 
            df['input_description'].fillna('') + " " + 
            df['output_description'].fillna('')
        )
        
        # 4. Target Encoding
        # Classification: Easy(0), Medium(1), Hard(2)
        class_map = {'easy': 0, 'medium': 1, 'hard': 2}
        df['label_cls'] = df['problem_class'].str.lower().map(class_map)
        
        # Regression: Score (Float)
        df['label_score'] = pd.to_numeric(df['problem_score'], errors='coerce')
        
        # Drop invalid rows (missing labels)
        df = df.dropna(subset=['label_cls', 'label_score', 'full_text'])
        
        return df
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return None

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print(f"✅ Data Ready: {len(df)} samples")
        print(df[['problem_class', 'label_cls', 'label_score']].head())

