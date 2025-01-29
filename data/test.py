import csv
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

def load_csv_to_tensordataset(filename):
    """Load data from CSV and convert it into a TensorDataset."""

    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Create the full file path
    file_path = os.path.join(script_directory, filename)

    # Load the CSV file using pandas
    df = pd.read_csv(file_path)
    
    # Process the "Number" column (convert to float if needed)
    numbers = df["Number"].apply(lambda x: float(str(x).replace('%', '').replace(' ', ''))).values

    # Tokenize the "Words" column (convert text to numerical labels or embeddings)
    label_encoder = LabelEncoder()
    words = label_encoder.fit_transform(df["Words"])  # Encode the words as integers
    
    # Convert both columns into tensors
    number_tensor = torch.tensor(numbers, dtype=torch.float32)  # Numbers as float tensors
    words_tensor = torch.tensor(words, dtype=torch.long)        # Words as integer labels
    
    # Create a TensorDataset
    dataset = TensorDataset(number_tensor, words_tensor)
    
    return dataset, label_encoder

# Example usage
if __name__ == "__main__":
    # Assuming the CSV file is saved as "random_numbers.csv"
    csv_filename = "random_numbers.csv"
    
    # Load data into TensorDataset
    dataset, label_encoder = load_csv_to_tensordataset(csv_filename)
    
    # Print a few examples from the dataset
    for i in range(5):  # Display the first 5 examples
        print(f"Tensor: {dataset[i][0]}, Label: {dataset[i][1]} (Word: {label_encoder.inverse_transform([dataset[i][1].item()])[0]})")
