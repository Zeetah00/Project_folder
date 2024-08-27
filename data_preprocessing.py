import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_data(file_paths):
    #a list to store the frames
    dfs = []
    
    #this loads each dataset and adds it to the list above
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        print(f"Columns in dataset from {file_path}:", df.columns)
        
        df.columns = df.columns.str.strip()
        
        #looks for the column label
        if 'Label' not in df.columns:
            raise KeyError(f"'Label' column not found in the dataset from {file_path}")

        #remove any row with a missing value
        df.dropna(inplace=True)

        #convert categorical labels to numbers
        df['Label'] = pd.factorize(df['Label'])[0]
        
        #replace dara with Nan if missing
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        #then add the processed data to the df list
        dfs.append(df)
    
    #add up all frames
    combined_df = pd.concat(dfs, ignore_index=True)
    X = combined_df.drop(['Label'], axis=1).values
    y = combined_df['Label'].values
    
    #normalizing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y
