import torch
import torch.nn as nn
from transformers import AutoModel

class AutoJudge(nn.Module):
    """
    Multi-task learning model for task complexity evaluation.
    Combines both classification (Easy/Medium/Hard) and regression (complexity score) heads.
    """
    def __init__(self, model_name="microsoft/deberta-v3-large", freeze_bert=False):
        super(AutoJudge, self).__init__()
        
        # Load Model (Safetensors to fix security error)
        self.bert = AutoModel.from_pretrained(model_name, use_safetensors=True)
        
        self.bert_dim = self.bert.config.hidden_size 
        
        # Processor layer (feature extraction)
        self.processor = nn.Sequential(
            nn.Linear(self.bert_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classification Head: Easy(0), Medium(1), Hard(2)
        self.classifier = nn.Linear(256, 3)
        
        # Regression Head: Complexity score (1.0 - 10.0)
        self.regressor = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output 0-1, scaled to 1-10 in forward
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask
            
        Returns:
            logits_cls: Classification logits (batch_size, 3)
            final_score: Regression score (batch_size, 1) scaled to [1.0, 10.0]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract [CLS] token representation (index 0)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        x = self.processor(cls_emb)
        
        logits_cls = self.classifier(x)
        raw_score = self.regressor(x)
        
        # Scale score from [0,1] to [1.0, 10.0]
        final_score = raw_score * 9.0 + 1.0 
        
        return logits_cls, final_score

