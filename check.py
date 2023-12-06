import pandas as pd

#datasets/standard/test/MSA_3Di_databases/tb_ssym.csv
#df = pd.read_csv("src/training_standard.csv", sep=',')
#df = df.sort_values(by=['code'])
#df.to_csv(f"src/training_standard_sort.csv", index=False)
df1 = pd.read_csv("datasets/standard/test/MSA_3Di_databases/tb_ssym.csv", sep=',')
df2 = pd.read_csv("datasets/standard/test/MSA_databases/db_ssym.csv", sep=',')
df2 = df2.sort_values(by=['code']).reset_index(drop=True)
df1 = df1.sort_values(by=['code']).reset_index(drop=True)
df1 = df1.drop(columns=['wt_struct', 'mut_struct'])
df2 = df2.drop(columns=['mut_info'])

print(df1.equals(df2))
for i in df2.code:
    if i not in df1.code.to_list():
        print(i)
#for a,b in zip(df1["code"], df2["code"]):
#    print(a,b)

#print(len(df2[df2.code=="1FEPA-Q256C"].wt_seq.to_list()[0]))
