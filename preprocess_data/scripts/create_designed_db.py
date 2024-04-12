
import pandas as pd
import random
import os
import shutil


mega = pd.read_csv('S157479/train/databases/db_s157479.csv')
small = pd.read_csv('S1465/train/databases/db_s1465.csv')


df = []
random.seed(0)
for p in list(set(mega['pdb_id'].to_list())):
    sel = mega[mega['pdb_id'] == p]
    indexes = sel.index.to_list()
    sel_index = random.sample(indexes, 15)
    df.append(mega.iloc[sel_index])
df = pd.concat(df)


db = pd.concat([small,df], axis=0)


train_path = 'S'+str(len(db))+'/train/'
if not os.path.exists(train_path):
    os.makedirs(train_path)
msa_path = train_path + 'MSA_S' +str(len(db))
if not os.path.exists(msa_path):
    os.makedirs(msa_path)
if not os.path.exists(train_path + 'databases'):
    os.makedirs(msa_path + 'databases')



for file in df['code'].to_list():
    shutil.copy2('S157479/train/MSA_S157479/'+file, msa_path)

for file in list(set(df['pdb_id'].to_list())):
    shutil.copy2('S157479/train/MSA_S157479/'+file, msa_path)

for file in small['code'].to_list():
    shutil.copy2('S1465/train/MSA_S1465/'+file, msa_path)


for file in list(set(small['pdb_id'].to_list())):
    shutil.copy2('S1465/train/MSA_S1465/'+file, msa_path)

db['mut_msa'] = [d.replace('S157479', 'S3730').replace('S1465','S3730') for d in db['mut_msa'].to_list()]

db['wt_msa'] = [d.replace('S157479', 'S3730').replace('S1465','S3730') for d in db['wt_msa'].to_list()]


db.to_csv(train_path + 'databases/db_s'+str(len(db))+'.csv')

