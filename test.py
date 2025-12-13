import pandas as pd
from agomax.pipeline import Pipeline

train_df = pd.read_csv("notebooks/train_normal.csv")

pipe = Pipeline(model_dir="models/")
pipe.fit(train_df)
