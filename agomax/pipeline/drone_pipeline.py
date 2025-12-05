import pandas as pd 
class DronePipeline:
    def __init__(self):
        pass
    
    def load(self,path):
        try:
            load_path = pd.read_csv(path)
            return load_path
        except FileNotFoundError:
            print('file not found',path)
        except Exception as e :
            print('Error loading CSV',e)
            
    