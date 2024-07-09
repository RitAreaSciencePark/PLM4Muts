<!--
SPDX-FileCopyrightText: 2024 (C) 2024 Marco Celoria <celoria.marco@gmail.com> and Francesca Cuturello <francesca.cuturello@areasciencepark.it>

SPDX-License-Identifier: CC-BY-4.0
-->

The code provided has been tested on the Booster partition of Leonardo, the pre-exascale Tier-0 EuroHPC supercomputer, at CINECA and on the DGX partition of Orfeo, the supercomputer hosted at AREA Science Park.

### Hardware requirements
Specifically, we tested the finetuning on the following architectures:

- Leonardo Booster one-node configuration:
	- Processors: single socket 32-core Intel Xeon Platinum 8358 CPU, 2.60GHz (Ice Lake)
	- RAM: 512 GB DDR4 3200 MHz 
	- Accelerators: 4x NVIDIA custom Ampere A100 GPU 64GB HBM2e, NVLink 3.0 (200GB/s)
	- Network: 2 x dual port HDR100 per node (400Gbps/node) 
	- All the nodes are interconnected through an Nvidia Mellanox network (Dragon Fly+).

- Orfeo DGX one-node configuration:
	- Processors: 2 x 64-core AMD EPYC 7H12 (2.6 GHz base, 3.3 GHz boost)
	- RAM: 1024 GB DDR4 3200 MT/s
	- Accelerators: 8x NVIDIA Ampere A100 SXM GPU 40GB HBM2e, NVLink 3.0 ?? (200GB/s)


### Software requirements

The software stack on Leonardo is as follows

```
$> srun --nodes=1 --ntasks-per-node=4 --cpus-per-task=8 --gres=gpu:4 -p boost_usr_prod --mem=450GB --time 02:50:00 --pty /bin/bash
$> module load python cuda nvhpc
```

Regarding the operating system, we tested the code with the following OS

```
$> cat /etc/os-release
NAME="Red Hat Enterprise Linux"
VERSION="8.7 (Ootpa)"
ID="rhel"
ID_LIKE="fedora"
VERSION_ID="8.7"
PLATFORM_ID="platform:el8"
PRETTY_NAME="Red Hat Enterprise Linux 8.7 (Ootpa)"
ANSI_COLOR="0;31"
CPE_NAME="cpe:/o:redhat:enterprise_linux:8::baseos"
HOME_URL="https://www.redhat.com/"
DOCUMENTATION_URL="https://access.redhat.com/documentation/red_hat_enterprise_linux/8/"
BUG_REPORT_URL="https://bugzilla.redhat.com/"

REDHAT_BUGZILLA_PRODUCT="Red Hat Enterprise Linux 8"
REDHAT_BUGZILLA_PRODUCT_VERSION=8.7
REDHAT_SUPPORT_PRODUCT="Red Hat Enterprise Linux"
REDHAT_SUPPORT_PRODUCT_VERSION="8.7"
```

As far as CUDA is concerned, we tested the code with the following configuration

```
$> nvidia-smi | grep CUDA 
| NVIDIA-SMI 530.30.02              Driver Version: 530.30.02    CUDA Version: 12.1     | 
```

Finally, we tested the code with the following python version
```
$> python3 --version
Python 3.11.6 
```

The dependencies are listed in the `requirements.txt` file.

### Installation 

The suggested procedure to install the dependencies on Leonardo is the following

```
$> module load python cuda nvhpc
$> python3 -m venv PLM4Muts_venv --system-site-packages
$> source PLM4Muts_venv/bin/activate
$> pip install -r requirements.txt 
```

On Leonardo, pre-trained weights can be downloaded only on the login node. 
Furthermore, the home directory is limited to 50GB per user.
For this reasons, we dowloaded the weights in the `src/models/models_cache/` directory using the `download_weights.job` and the `src/download_weights.py` scripts.


### Data Format and Structure


Data must be in csv format. The following columns must be complete and specified in the header:

- 'pdb_id', 'code', 'pos', 'wt_seq', 'mut_seq', 'wt_msa', 'ddg'

For training, create a directory associated with the experiment in `datasets/train_name` containing the following subdirectories: **train**, **test** and **validation**. 

Each subdir must contain the `database/db_name.csv` and `MSA_train_name` directory with the wild type MSA.

For the inference only the test set is needed.


### Execution

For training, associate the experiments with a directory for example `runs/experiment_name`. 
This must contain:
- the script to launch the program (similar to the `finetuning.job` for systems with Slurm scheduler) 
- the `config.yaml` file reporting the paths and parameters specific to the run. 

The following example to show the required fields in this `config.yaml`

```
output_dir: "runs/experiment_name"
dataset_dir: "datasets/train_name"
model: "MSA_Finetuning"
learning_rate: 1.0e-4
max_epochs: 20
loss_fn: "L1"
max_length: 1024
seeds: [10, 11, 12]
optimizer:
  name: "AdamW"
  weight_decay: 0.01
  momentum: 0.
MSA:
  max_tokens: 16000
snapshot_file: "runs/ experiment_name/snapshots/MSA_Finetuning.pt"
```

The possible options for the model field are: 

- `MSA_Finetuning`
- `ESM2_Finetuning`
- `PROST5_Finetuning`
- `MSA_Baseline`
- `ESM2_Baseline`
- `PROST5_Baseline`

The possible options for the `loss_fn` field are:

- "L1"
- "MSE"

The `max_length` parameter regulates the max length of the sequences that can be loaded in memory.
That is, sequences with length greater than `max_length` are discarded.

Seeds consists of a list of three integers. The final seed per GPU is computed according to:

`seed = seeds[0] * (seeds[1] + seeds[2] * GPU_rank)`

The possible options for the optimizer field are:

- "Adam"
- "AdamW"
- "SGD"

For `AdamW` and `SGD` one can also specify the `weight_decay` parameter.

For `SGD` one can eventually specify the `momentum` parameter.

The `max_tokens` parameters is effective only for the MSA case and specifies the max number of tokens that can be loaded in memory (that is `length x depth`).

Too large values of `max_tokens` can result in memory issues such as `CUDA_MEMORY_ERROR`.   

Outputs can be found in `runs/experiment_name/results/` and consists of the following files:

- `db_test_name.res`: selected epoch, rmse, mae, corr, p-value for the test set

- `db_test_name_labels_preds.diffs`: predicted and experimental values for each sequence in test set

- `epochs_statistics.csv`: summary of results at all epochs for test, validation and training

- `early_stopping_epoch.log`: selected epoch

- `test_db_test_name_metrics.log`: rmse, mae, corr on test set for all training epochs

- `train_db_train_name_metrics.log`: rmse, mae, corr on cross-validated training set for all training epochs

- `val_db_val_name_metrics.log`: rmse, mae, corr on validation set for all training epochs

- `seeds.log`: seeds parameter used for the run

- `epochs_rsme.png`

- `epochs_corr.png`

- `epochs_mae.png`


### Inference

In order to perform an Inference have a look at the `runs/Inference_MSA_Finetuning` and `datasets/Inference` folders.

As an example, we provide an example dataset, where we consider 13 mutations of the 1A7V protein.
Dataset files correctly organized as follows

```bash
datasets/Inference/
`-- test
    |-- databases
    |   `-- db_s13.csv
    |-- MSA_s13
    |   `-- 1A7V
    `-- translated_databases
        `-- tb_s13.csv
```
We have generated `translated_databases/tb_s13.csv` by means of the `src/ProstT5TranslationDDP.py` program (see for instance `runs/S1465_Translate/translateS1465.sh` for more details). 

Now, in `runs/Inference_MSA_Finetuning` we have a `config.yaml` file where we specify for the MSA model

```
output_dir: "runs/Inference_MSA_Finetuning"
dataset_dir: "datasets/Inference"
model: "MSA_Finetuning"
max_length: 1024
MSA:
  max_tokens: 16000
snapshot_file: "runs/S1413_MSA_Finetuning/snapshots/MSA_Finetuning.pt"
```

To perform the inference we provide a slurm job template in `runs/Inference_MSA_Finetuning/inference.job`, to be adjusted in accordance to your needs.
 

