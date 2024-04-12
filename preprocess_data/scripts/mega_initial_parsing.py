import pandas as pd

df = pd.read_csv('training_data/mega_original.csv', index_col=0)

df = df[df['reverse']==False]

df['pdb_id'] = [x.replace('.pdb','') for x in df['pdb_id'].to_list()]

df.to_csv('training_data/mega.csv', index=False)

