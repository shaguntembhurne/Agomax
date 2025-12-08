import pandas as pd 
class DronePipeline:
    def __init__(self):
        self.data  = None 
        
    def load(self,path):
        try:
            self.data = pd.read_csv(path)
            return self.data
        except FileNotFoundError:
            print('file not found',path)
        except Exception as e :
            print('Error loading CSV',e)
    def preprocess(self):
        df = self.data
        
        # Step 1: Fill NaN values for numeric columns
        for col in df.columns:
            
            # check numeric
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"{col} is not numeric")
                continue
            
            # fill NaN
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        
        # Step 2: Standard scaling for ALL numeric columns
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

        return df
        