```
srun --nodes=1 --ntasks-per-node=1 -A lade --mem=40GB --ntasks-per-socket=1  --sockets-per-node=1 --gpus=1 --gpus-per-node=1 --gpus-per-socket=1 --cpus-per-task=32 -p DGX   -w dgx002 --time=4:01:00  --pty bash
```

```
$> cat /etc/os-release

NAME="Ubuntu"
VERSION="20.04.6 LTS (Focal Fossa)"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 20.04.6 LTS"
VERSION_ID="20.04"
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
VERSION_CODENAME=focal
UBUNTU_CODENAME=focal
```

```
$> nvidia-smi | grep CUDA 

| NVIDIA-SMI 525.147.05   Driver Version: 525.147.05   CUDA Version: 12.0     |
```

```
$> python3 --version

Python 3.8.10
```

```
$> python3 -m venv PLM4Muts_venv
$> source PLM4Muts_venv/bin/activate
$> pip install -r requirements.txt  
```


