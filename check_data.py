import pandas as pd

df = pd.read_csv('data/training_chunk.csv')
print("Sample of training data:")
print(df.head(20))
print(f"\nis_circular rate: {df['is_circular'].mean():.4%}")
print(f"Total positives: {df['is_circular'].sum()}")
