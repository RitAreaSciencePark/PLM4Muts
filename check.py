import pandas as pd

#datasets/standard/test/MSA_3Di_databases/tb_ssym.csv
#df = pd.read_csv("src/training_standard.csv", sep=',')
#df = df.sort_values(by=['code'])
#df.to_csv(f"src/training_standard_sort.csv", index=False)

#datasets/S1827/train/databases/db_train.csv datasets/S1827/train/translated_databases/tb_train.csv

df1a = pd.read_csv("datasets/S1618/train/databases/db_train.csv", sep=',')
df2a = pd.read_csv("datasets/S1618/train/translated_databases/tb_train.csv", sep=',')
df3a = pd.read_csv("datasets/S1618/validation/databases/db_ssym.csv")
df4a = pd.read_csv("datasets/S1618/validation/translated_databases/tb_ssym.csv")
df5a = pd.read_csv("datasets/S1618/test/databases/db_s669.csv")
df6a = pd.read_csv("datasets/S1618/test/translated_databases/tb_s669.csv")

#df1b = pd.read_csv("datasets/S3120/train/databases/db_train.csv", sep=',')
#df2b = pd.read_csv("datasets/S3120/train/translated_databases/tb_train.csv", sep=',')
#df3b = pd.read_csv("datasets/S3120/test/databases/db_s669.csv")
#df4b = pd.read_csv("datasets/S3120/test/translated_databases/tb_s669.csv")
#df5b = pd.read_csv("datasets/S3120/validation/databases/db_ssym.csv")
#df6b = pd.read_csv("datasets/S3120/validation/translated_databases/tb_ssym.csv")

df1a = df1a.sort_values(by=['code']).reset_index(drop=True)
df2a = df2a.sort_values(by=['code']).reset_index(drop=True)
df3a = df3a.sort_values(by=['code']).reset_index(drop=True)
df4a = df4a.sort_values(by=['code']).reset_index(drop=True)
df5a = df5a.sort_values(by=['code']).reset_index(drop=True)
df6a = df6a.sort_values(by=['code']).reset_index(drop=True)
#df1b = df1b.sort_values(by=['code']).reset_index(drop=True)
#df2b = df2b.sort_values(by=['code']).reset_index(drop=True)
#df3b = df3b.sort_values(by=['code']).reset_index(drop=True)
#df4b = df4b.sort_values(by=['code']).reset_index(drop=True)
#df5b = df5b.sort_values(by=['code']).reset_index(drop=True)
#df6b = df6b.sort_values(by=['code']).reset_index(drop=True)

df2a = df2a.drop(columns=['wt_struct', 'mut_struct'])
df4a = df4a.drop(columns=['wt_struct', 'mut_struct'])
df6a = df6a.drop(columns=['wt_struct', 'mut_struct'])

#df2b = df2b.drop(columns=['wt_struct', 'mut_struct'])
#df4b = df4b.drop(columns=['wt_struct', 'mut_struct'])
#df6b = df6b.drop(columns=['wt_struct', 'mut_struct'])


#df2 = df2.drop(columns=['mut_info'])
print(len(df1a), len(df2a))
print(len(df3a), len(df4a))
print(len(df5a), len(df6a))

#print(len(df1b), len(df2b))
#print(len(df3b), len(df4b))
#print(len(df5b), len(df6b))
#print(df1a.equals(df2a))

for i in df1a.code:
    if i not in df2a.code.to_list():
        print(i)

#for a,b in zip(df1["code"], df2["code"]):
#    print(a,b)

#print(len(df2[df2.code=="1FEPA-Q256C"].wt_seq.to_list()[0]))


df1 = pd.read_csv("runs/run_S1618_MSA_HP001/results/tb_s669_AAA.diffs")
df2 = pd.read_csv("runs/run_S1618_MSA_HP001/results/tb_s669_labels_preds.diffs")
print(len(df1), len(df2), len(set(df1.code.to_list())))
seen={}
for i in df1.code:
    if i in seen:
        seen[i]+=1
    else:
        seen[i]=0
print(df1.pred.sum(), df2.pred.sum())
for i,j in seen.items():
    if j==1:
        print(i,j)

