
import pandas as pd
import os
import numpy as np

# Create directory
if not os.path.exists('test_data'):
    os.makedirs('test_data')

try:
    # Read data
    df = pd.read_csv('winequality.csv')
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into 5 parts
    chunks = np.array_split(df, 5)
    
    for i, chunk in enumerate(chunks):
        output_file = f'test_data/wine_test_batch_{i+1}.csv'
        chunk.to_csv(output_file, index=False)
        print(f"Created {output_file} with {len(chunk)} rows.")
        
    print(f"Successfully created {len(chunks)} test files in 'test_data/' directory.")

except FileNotFoundError:
    print("Error: winequality.csv not found.")
except Exception as e:
    print(f"An error occurred: {e}")
