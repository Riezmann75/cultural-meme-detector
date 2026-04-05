import os
import shutil
import random
from pathlib import Path

def split_dataset(input_root, output_root, split_ratio=(0.8, 0.1, 0.1), seed=42):
    """
    Splits the dataset into train, val, and test sets.
    Expected structure: input_root/{lang}/{positive/negative}/images.jpg
    Output structure: output_root/{train/val/test}/{lang}/{positive/negative}/images.jpg
    """
    random.seed(seed)
    
    # Define the splits
    split_names = ['train', 'val', 'test']
    
    # Get all language folders (e.g., 'id', 'vi')
    languages = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    
    for lang in languages:
        lang_path = os.path.join(input_root, lang)
        # Handle typos in folder names (e.g., 'postitive' in your screenshot)
        classes = [d for d in os.listdir(lang_path) if os.path.isdir(os.path.join(lang_path, d))]
        
        for cls in classes:
            cls_path = os.path.join(lang_path, cls)
            images = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
            random.shuffle(images)
            
            # Calculate split indices
            n = len(images)
            train_end = int(n * split_ratio[0])
            val_end = train_end + int(n * split_ratio[1])
            
            splits = {
                'train': images[:train_end],
                'val': images[train_end:val_end],
                'test': images[val_end:]
            }
            
            for split_name, split_images in splits.items():
                # Target path: output/train/id/positive/
                target_dir = os.path.join(output_root, split_name, lang, cls)
                os.makedirs(target_dir, exist_ok=True)
                
                for img_name in split_images:
                    src_path = os.path.join(cls_path, img_name)
                    dst_path = os.path.join(target_dir, img_name)
                    shutil.copy2(src_path, dst_path)
                    
        print(f"Finished splitting language: {lang}")

if __name__ == "__main__":
    # CONFIGURATION
    INPUT_FOLDER = "dataset" # The folder in your screenshot
    OUTPUT_FOLDER = "split_dataset"
    SPLITS = (0.8, 0.1, 0.1) # Train, Val, Test
    
    # Run the split
    if os.path.exists(INPUT_FOLDER):
        split_dataset(INPUT_FOLDER, OUTPUT_FOLDER, split_ratio=SPLITS)
        print(f"\nSuccess! Your split dataset is located in: '{OUTPUT_FOLDER}'")
    else:
        print(f"Error: Could not find folder '{INPUT_FOLDER}'")
