# AutoJudge: Task Complexity Evaluator ⚖️

## NOTE :
The model's weights have not been provided here , you can get them by running ./regressor_classifier_train_and_run.ipynb , you will not be able to host on streamlit otherwise .
Model weights if you need access beforehand are available in a public notebook output here at kaggle : https://www.kaggle.com/code/abhirajbharangar/autojudge-modern-llm/output?scriptVersionId=290720138
Plese maintain proper naming , preffered weights in ./model_weights/autojudge_best_acc_model.pth naming maintain for all functioning , check versions for different models , version 1 is currently paired with best accuracy , ./regressor_classifier_train_and_run.ipynb
## 1. Project Overview

AutoJudge is an AI-powered system designed to automatically evaluate the complexity of programming tasks. By analyzing problem statements, input/output descriptions, and constraints, the model predicts:

1. **Difficulty Class**: (Easy, Medium, Hard) - Classification task
2. **Complexity Score**: A continuous value from 1.0 to 10.0 - Regression task

This project aims to assist educators and competitive programming platforms in categorizing problems efficiently without manual review, leveraging state-of-the-art NLP techniques with Multi-Task Learning (MTL).

**Note**: ( first download the weights from kaggle link above and maintain the structure mentioned here , check versions for different models) The model combines both classification and regression into a single unified architecture, sharing the same backbone encoder and using separate heads for each task The Best model is at ./regressor_classifier_model_train_and_run.ipynb , with its weights at ./model_weights/autojudge_best_acc_model.pth other models with if you want to run with just a different model , with analogous code they are inside ./other_models_regressor_and_classifier_train_and_run/{model_name}.ipynb , and their weights are in ./model_weights/{model_name}_weight.pth .
 
## Structure overview :
    •  Data preprocessing : ./preprocess_extract_dataset.ipynb or ./src/data/data_loader.py both are same .
    •  Feature extraction : included in ./preprocess_extract_dataset.ipynb or ./src/data/data_loader.py
    •  Classification model (Easy / Medium / Hard) and Regressor model : In ./regressor_classifier_model_train_and_run.ipynb , weights in ./model_weights/autojudge_best_acc_model.pth
    •  Web UI code (Flask / Streamlit / etc.) : in ./src/web/streamlit_app.py , to run streamlit app write in terminal: streamlit run src/web/streamlit_app.py or Double-click run_app.bat in the project root folder , guide in ./STREAMLIT_GUIDE.md



---

## 2. Dataset

- **Source**: [TaskComplexityEval-24](https://github.com/AREEG94FAHAD/TaskComplexityEval-24)
- **Size**: ~4,112 programming tasks scraped from platforms like Kattis, LeetCode, etc.
- **Location**: `data/problems_data.jsonl` (automatically downloaded if not present)

**Dataset Structure**:
- `title`: Problem title
- `description`: Main problem statement
- `input_description`: Input constraints and format
- `output_description`: Output format specifications
- `problem_class`: Ground truth class (easy/medium/hard)
- `problem_score`: Ground truth complexity score (1.1 - 9.9)

**Data Preprocessing** (see `src/data/data_loader.py`):
- All text fields are concatenated with `[SEP]` separators to create `full_text` feature
- Classification labels mapped: easy→0, medium→1, hard→2
- Regression scores normalized to float values

---

## 3. Approach and Models Used

### Architecture

We employed a **Multi-Task Learning (MTL)** approach using **Microsoft DeBERTa-v3-Large** as the backbone architecture.

**Model Components**:

1. **Backbone Encoder**: 
   - `microsoft/deberta-v3-large` (Shared encoder)
   - Pre-trained on large-scale text data
   - Generates contextual embeddings from input text

2. **Optimization**: 
   - **LoRA (Low-Rank Adaptation)** applied for efficient fine-tuning
   - Targets: `query_proj`, `key_proj`, `value_proj`, `dense` layers
   - Hyperparameters: `r=16`, `lora_alpha=32`, `dropout=0.1`
   - Enables training on consumer hardware with reduced memory footprint

3. **Task Heads**:
   - **Feature Processor**: Linear layer (1024 → 256) with ReLU and Dropout(0.2)
   - **Classifier Head**: Linear layer (256 → 3) for difficulty classification
   - **Regressor Head**: Linear layer (256 → 1) with Sigmoid, scaled to [1.0, 10.0]

### Training Details

- **Precision**: Mixed Precision Training (FP16) using `torch.cuda.amp`
- **Loss Function**: Combined `CrossEntropyLoss` (Classification) + `MSELoss` (Regression)
- **Optimizer**: `AdamW` with `lr=1e-4` and `weight_decay=0.1`
- **Scheduler**: Linear warmup scheduler (10% warmup steps)
- **Batch Size**: 8 (optimized for GPU memory)
- **Max Sequence Length**: 350 tokens
- **Training Strategy**: Early stopping with patience=3 based on validation accuracy
- **Epochs**: Up to 10 (with early stopping)

### Feature Extraction

- Text preprocessing combines: `title + [SEP] + description + [SEP] + input_description + output_description`
- Tokenization using DeBERTa-v3 tokenizer
- [CLS] token embedding used as the final representation

---

## 4. Evaluation Metrics

The model was evaluated on a 20% held-out test set using the best model saved based on classification accuracy.

| Metric | Score |
|:---|:---|
| **Classification Accuracy** | **[INSERT FINAL ACCURACY]%** |
| **Regression MAE (Mean Absolute Error)** | **[INSERT FINAL MAE]** |
| **Regression RMSE (Root Mean Squared Error)** | **[INSERT FINAL RMSE]** |

*(Metrics are calculated on the final evaluation run on the test split after training completion)*

---

## 5. Steps to Run the Project Locally

### Prerequisites

- **Python**: 3.8 or higher
- **CUDA-enabled GPU**: Recommended for faster inference (CPU supported but slower)
- **Memory**: At least 8GB GPU VRAM recommended for training, 4GB+ for inference

### Installation

1. **Clone the repository**:
   ```bash
   git clone []
   cd AutoJudge_Project_24117002
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Model Weights**:
   - Ensure the trained model file `autojudge_best_acc_model.pth` is located in the `model_weights/` directory
   - If you haven't trained yet, follow the training steps below

### Training the Model

. **Open and run the training notebook**:
   - Open `regressor_classifier_model_train_and_run.ipynb` in Jupyter/VS Code
   - The notebook will:
     - Automatically download the dataset to `../data/problems_data.jsonl`
     - Split data into train/test (80/20)
     - Train the model with LoRA fine-tuning
     - Save the best model to `../model_weights/autojudge_best_acc_model.pth`
     - Display final evaluation metrics

   **Note**: Training may take several hours depending on your GPU. The model uses early stopping to prevent overfitting.

### Running the Web Interface

**Quick Start** (Windows):
- Double-click `run_app.bat` in the project root

**Manual Start**:
1. **From the project root directory**, run:
   ```bash
   streamlit run src/web/streamlit_app.py
   ```

2. **Access the application**:
   - The app will automatically open in your browser at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

3. **Using the Interface**:
   - Enter problem details in the left panel
   - Click "🚀 Analyze Complexity" to get predictions
   - View results in the right panel (Difficulty class + Complexity score)

**📖 Detailed Guide**: See `STREAMLIT_GUIDE.md` for:
- Step-by-step hosting instructions
- Troubleshooting tips
- Demo video creation guide with sample problems

---

## 6. Project Structure

```
AutoJudge_Project_24117002/
├── src/                          # Source code
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py       # Data preprocessing and loading
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py             # Model architecture definition
│   └── web/
│       └── streamlit_app.py     # Streamlit web interface
├── notebooks/                    # Jupyter notebooks
│   ├── Train_and_run_regressor_and_classifier.ipynb  # Main training notebook
│   └── dataset.ipynb            # Dataset exploration notebook
├── data/                         # Data directory
│   └── problems_data.jsonl      # Dataset (auto-downloaded)
├── model_weights/                       # Model weights
│   └── autojudge_best_acc_model.pth  # Trained model checkpoint
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 7. Explanation of the Web Interface

The Streamlit web application provides a user-friendly interface for task complexity evaluation.

### Input Fields

- **Task Title**: Name or title of the programming problem
- **Problem Description**: Main problem statement (required)
- **Additional Details (Optional)**:
  - Input Constraints: e.g., "N <= 10^5"
  - Output Format: e.g., "Print a single integer"

### Output Display

1. **Difficulty Classification**:
   - Visual indicator with color coding:
     - 🟢 Green: Easy
     - 🟡 Yellow: Medium
     - 🔴 Red: Hard

2. **Complexity Score**:
   - Numerical value between 1.0 and 10.0
   - Progress bar visualization showing relative complexity

3. **Technical Debug Info** (Expandable):
   - Raw classification logits for each class
   - Exact regression score
   - Input token count

### How It Works

1. User inputs problem details
2. Text is formatted: `title [SEP] description [SEP] input_constraints output_format`
3. Tokenization using DeBERTa-v3 tokenizer
4. Model inference (classification + regression)
5. Results displayed with visual indicators

---

## 8. Demo Video

**[https://drive.google.com/file/d/1hLyrOIZDFwqaW6CssQRFXUfmJQw9xCtS/view?usp=drive_link]**

*The demo video should showcase:*
- Dataset overview
- Model training process
- Web interface usage
- Example predictions on different problem types

---

## 9. Author Information

**Name**: Abhiraj Bharangar  
**Branch**: Computer Science and Engineering, 2nd Year B.Tech  
**Enrollment Number**: 24117002

---

## 10. License

See `LICENSE` file for details.

---

## 11. Acknowledgments

- Dataset: [TaskComplexityEval-24](https://github.com/AREEG94FAHAD/TaskComplexityEval-24)
- Model: [Microsoft DeBERTa-v3-Large](https://huggingface.co/microsoft/deberta-v3-large)
- LoRA Implementation: [PEFT Library](https://github.com/huggingface/peft)

---

## 12. Notes

- The model combines both classification and regression tasks in a single architecture (not separate models)
- All weights are saved in a single `.pth` file: `./model_weights/autojudge_best_acc_model.pth`
- The model uses LoRA for efficient fine-tuning, reducing memory requirements
- For best results, ensure input text follows the same format as training data
