import pdb

import torch.nn as nn
from transformers import SiglipVisionModel
import torch

class SiglipMemeClassifier(nn.Module):
    def __init__(self, model_name="google/siglip2-base-patch16-256", dropout_rate=0, tune_pos_embed=False):
        super().__init__()
        # 1. Vision Backbone
        self.vision_tower = SiglipVisionModel.from_pretrained(model_name)
        hidden_size = self.vision_tower.config.hidden_size
        
        # --- STRATEGIC UNFREEZING ---
        # 1. Freeze everything first (including all encoder blocks)
        for param in self.vision_tower.parameters():
            param.requires_grad = False
            
        # 2. Unfreeze the BUILT-IN Attention Pooling Head ONLY
        # This is the exact module you referenced: SiglipMultiheadAttentionPoolingHead
        for param in self.vision_tower.vision_model.head.parameters():
            param.requires_grad = True
        print("[Model] Unfrozen built-in SigLIP Attention Pooling Head")
            
        # 3. Unfreeze Position Embeddings (Optional but recommended for layout understanding)
        if tune_pos_embed:
            self.vision_tower.vision_model.embeddings.position_embedding.requires_grad = True
            print("[Model] Unfrozen Position Embeddings")
        # ----------------------------
        
        # Unfreeze last encoder block (Optional, can be skipped if you want to be more conservative)
        for param in self.vision_tower.vision_model.encoder.layers[-1].parameters():
            param.requires_grad = True
        print("[Model] Unfrozen last encoder block of the vision tower")

        # 2. Final Binary Classifier
        # Because the built-in head already contains the heavy 3072-D MLP and LayerNorm,
        # we only need a simple classifier on top of its output.
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        # The vision tower automatically routes the patches through the 
        # built-in SiglipMultiheadAttentionPoolingHead internally.
        outputs = self.vision_tower(pixel_values=x)
        
        # This pooler_output IS the output of the head you wanted to fine-tune!
        pooled_features = outputs.pooler_output
        
        # Pass the pooled features to our 2-class binary head
        return self.classifier(pooled_features)


if __name__ == "__main__":
    model = SiglipMemeClassifier()
    pdb.set_trace()
    dummy_input = torch.randn(4, 3, 256, 256)  # batch of 4 images
    output = model(dummy_input)
    print(output.shape)  # should be [4, 2]
