import torch.nn as nn
from transformers import SiglipVisionModel

class SiglipMemeClassifier(nn.Module):
    def __init__(self, model_name="google/siglip-base-patch16-224", dropout=0.1):
        super().__init__()
        # Load pre-trained SigLIP vision tower
        self.vision_tower = SiglipVisionModel.from_pretrained(model_name)
        # Freeze vision tower (optional, set to True if dataset is small)
        for param in self.vision_tower.parameters():
            param.requires_grad = False
            
        # Classification Head
        hidden_size = self.vision_tower.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2) # Binary: 0 or 1
        )

    def forward(self, x):
        outputs = self.vision_tower(pixel_values=x)
        # Use the pooled output (embedding of the whole image)
        features = outputs.pooler_output 
        return self.classifier(features)
