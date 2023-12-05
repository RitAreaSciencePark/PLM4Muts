import os
path = '/orfeo/scratch/dssc/mceloria/PLM4Muts/datasets/mega/train/MSA_mega'
files = os.listdir(path)


for index, file in enumerate(files):
    a=file.split('-')
    if len(a)==3:
        new_filename=a[1]+"-"+a[2]
        #print(os.path.join(path, file), os.path.join(path, new_filename))
        os.rename(os.path.join(path, file), os.path.join(path, new_filename))
