srun --nodes=1 --ntasks-per-node=1 -A lade --ntasks-per-socket=1  --sockets-per-node=1 --gpus=1 --gpus-per-node=1 --gpus-per-socket=1 --cpus-per-task=32 -p DGX   -w dgx002 --time=4:01:00  --pty bash

python -m venv myenv_dgx

source myenv_dgx/bin/activate

pip3 install torch torchvision torchaudio transformers sentencepiece accelerate --extra-index-url https://download.pytorch.org/whl/cu118
pip3 install matplotlib
pip3 install pandas
pip3 install scipy
pip install protobuf
