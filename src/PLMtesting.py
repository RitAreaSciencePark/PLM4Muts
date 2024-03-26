import os
from pathlib import Path
from models.models import *
from dataloader  import *
from utils  import *
from argparser import *
from tester import *


def main(output_dir, dataset_dir, model_name, max_length, max_tokens, snapshot_file):
    test_dir = dataset_dir + "/test"

    if model_name.rsplit("_")[0]=="ProstT5":
        test_dfs, test_names = from_cvs_files_in_dir_to_dfs_list(test_dir, datasets_dir="/translated_databases")
        test_dss = [ProstT5_Dataset(df=test_df,name=test_name,max_length=max_length) for test_df, test_name in zip(test_dfs, test_names)]
        collate_function = None
    if model_name.rsplit("_")[0]=="ESM2":
        test_dfs, test_names = from_cvs_files_in_dir_to_dfs_list(test_dir, datasets_dir="/databases")
        test_dss = [ESM2_Dataset(df=test_df,name=test_name,max_length=max_length) for test_df, test_name in zip(test_dfs, test_names)]
        collate_function = custom_collate
    if model_name.rsplit("_")[0]=="MSA":
        test_dfs, test_names = from_cvs_files_in_dir_to_dfs_list(test_dir, datasets_dir="/databases")
        test_dss = [MSA_Dataset(df=test_df,name=test_name, dataset_dir=test_dir, max_length=max_length,
                                max_tokens=max_tokens) for test_df, test_name in zip(test_dfs, test_names)] 
        collate_function = custom_collate

    test_dls = [ProteinDataLoader(test_ds, batch_size=1, num_workers=0, shuffle=False, pin_memory=False,sampler=None,custom_collate_fn=collate_function) for test_ds in test_dss]
    print(f"output_dir:\t{output_dir}\t{type(output_dir)}", flush=True)
    print(f"model_name:\t{model_name}\t{type(model_name)}", flush=True)
    print(f"snapshot_file:\t{snapshot_file}\t{type(snapshot_file)}", flush=True)
    print(f"test_dir:\t{test_dir}\t{type(test_dir)}", flush=True)
    print(f"max_length:\t{max_length}\t{type(max_length)}", flush=True)
    print(f"max_tokens:\t{max_tokens}\t{type(max_tokens)}", flush=True)
    tester     = Tester(output_dir=output_dir)
    test_model = models[model_name]()
    tester.test(test_model=test_model, test_dls=test_dls, snapshot_file=snapshot_file)

if __name__ == "__main__":
    args = argparser_tester()
    config_file = args.config_file
    config_path = Path(config_file)
    if not os.path.exists(config_path):
        print(f"The path {config_path} doesn't exist")

    if os.path.isfile(config_path):
        print(f"Opening the configuration file {config_file}")
        config = load_config(config_file)
        try:
            dataset_dir = config["dataset_dir"]
            dataset_path = Path(dataset_dir)
            if not os.path.exists(dataset_path):
                print(f"The dataset path {dataset_path} doesn't exist")
                raise SystemExit(1)
            if not os.path.isdir(dataset_path):
                print(f"The dataset path {dataset_path} is not a directory")
                raise SystemExit(1)
        except:
            print(f"The dataset directory doesn't exist")
            raise SystemExit(1)

        try:
            snapshot_file = config["snapshot_file"]
            snapshot_path = Path(snapshot_file)
            if not os.path.exists(snapshot_path):
                print(f"The snapshot path {snapshot_path} doesn't exist")
                raise SystemExit(1)
            if not os.path.isfile(snapshot_path):
                print(f"The snapshot path {snapshot_path} is not a file")
                raise SystemExit(1)
        except:
            print(f"The snapshot file doesn't exist")
            raise SystemExit(1)

        try:
            output_dir = config["output_dir"]
        except:
            output_dir = os.getcwd()
            print(f"Setting the default output directory: {output_dir}")
        try:    
            model_name = config["model"]
        except:
            print(f"Error while setting the model in config file {config_file}")
            raise SystemExit(1)
        try:
            max_length = config["max_length"]
        except:
            max_length = 1024
            print(f"Setting the default max length of the aminoacid sequence: {max_length}")
        try:
            max_tokens = config["MSA"]["max_tokens"]
        except:
            max_tokens = 16000
            if model_name == "MSA_Finetuning" or model_name == "MSA_Baseline":
                print(f"Setting the default max number of tokens for MSA: {max_tokens}")
    else:
        print(f"The path {config_path} is not valid")
        raise SystemExit(1)

    main(output_dir, dataset_dir, model_name, max_length, max_tokens, snapshot_file)


