import pandas as pd

def read_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']]
    return df.astype(float).values.tolist()
