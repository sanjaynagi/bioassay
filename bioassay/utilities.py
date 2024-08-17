import pandas as pd
import os

def load_bioassay_data(file_path):
    """
    Load bioassay data from Excel, CSV, or TSV files into a pandas DataFrame.
    
    Parameters:
    file_path (str): Path to the input file
    
    Returns:
    pandas.DataFrame: Loaded data
    """
    # Get the file extension
    _, file_extension = os.path.splitext(file_path.lower())
    
    try:
        if file_extension in ['.xlsx', '.xls']:
            # Load Excel file
            df = pd.read_excel(file_path)
        elif file_extension == '.csv':
            # Load CSV file
            df = pd.read_csv(file_path)
        elif file_extension == '.tsv':
            # Load TSV file
            df = pd.read_csv(file_path, sep='\t')
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        return df
    
    except Exception as e:
        print(f"Error loading file: {e}")
        return None