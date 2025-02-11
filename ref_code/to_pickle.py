import pandas as pd
import joblib

csv_path = 'train.csv'

df = pd.read_csv(csv_path, nrows=0)
column_names = df.columns.tolist()

joblib_file_path = 'cat_cols.pickle'
joblib.dump(column_names, joblib_file_path)