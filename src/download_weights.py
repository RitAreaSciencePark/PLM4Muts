import sys
import os
sys.path.append("PLM4Muts_venv/lib/python3.11/site-packages/")
os.environ["TRANSFORMERS_CACHE"] = "./src/models/models_cache"
from models.models import *
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM

print(esm.__version__)

torch.hub.set_dir("./src/models/models_cache")
esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
msa_model, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
prostt5_encoder  = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
prostt5_alphabet = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
prostt5_seq2seq  = AutoModelForSeq2SeqLM.from_pretrained("Rostlab/ProstT5")

#prostt5_encoder.save_pretrained ("./src/models/ProstT5")
#prostt5_alphabet.save_pretrained("./src/models/ProstT5")
#prostt5_seq2seq.save_pretrained ("./src/models/ProstT5")
