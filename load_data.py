import os
import pandas as pd
from PIL import Image
import numpy as np
from pathlib import Path

def load_image_data(data_dir='data'):
    image_data = []
    
    # assumes data directory is data/ by default
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            # Check if file is a JPEG image
            if file.lower().endswith(('.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                
                #use path to determine label and split
                path_parts = Path(file_path).parts
                
                if 'NORMAL' in path_parts:
                    label = 'NORMAL'
                elif 'PNEUMONIA' in path_parts:
                    label = 'PNEUMONIA'
                else:
                    label = 'UNKNOWN'
                
                if 'train' in path_parts:
                    split = 'train'
                elif 'test' in path_parts:
                    split = 'test'
                elif 'val' in path_parts:
                    split = 'val'
                else:
                    split = 'unknown'
                
                # load image data
                try:
                    img = Image.open(file_path)
                    img_array = np.array(img)
                    
                    image_data.append({
                        'image_path': file_path,
                        'label': label,
                        'split': split,
                        'image_array': img_array,
                        'shape': img_array.shape,
                        'filename': file
                    })
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    df = pd.DataFrame(image_data)
    
    return df


if __name__ == "__main__":
    print("Loading image data...")
    df = load_image_data()
    pickle_filename = 'image_data.pkl'
    df.to_pickle(pickle_filename)
    print(f"Saved dataframe as .pkl file to {pickle_filename}")
