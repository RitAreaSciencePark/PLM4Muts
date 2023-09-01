from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)

# Load the model
model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)

# only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
model.full() if device=='cpu' else model.half()

# prepare your protein sequences/structures as a list. Amino acid sequences are expected to be upper-case ("PRTEINO" below) while 3Di-sequences need to be lower-case ("strctr" below).
sequence_examples = ["PRTEINO", "strct"]

# replace all rare/ambiguous amino acids by X (3Di sequences does not have those) and introduce white-space between all sequences (AAs and 3Di)
sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

# add pre-fixes accordingly (this already expects 3Di-sequences to be lower-case)
# if you go from AAs to 3Di (or if you want to embed AAs), you need to prepend "<AA2fold>"
# if you go from 3Di to AAs (or if you want to embed 3Di), you need to prepend "<fold2AA>"
sequence_examples = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s
                      for s in sequence_examples
                    ]

# tokenize sequences and pad up to the longest sequence in the batch
ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest",return_tensors='pt').to(device)

# generate embeddings
with torch.no_grad():
    embedding_repr = model(
              ids.input_ids, 
              attention_mask=ids.attention_mask
              )

# extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens, incl. prefix ([0,1:8]) 
emb_0 = embedding_repr.last_hidden_state[0,1:8] # shape (7 x 1024)
# same for the second ([1,:]) sequence but taking into account different sequence lengths ([1,:6])
emb_1 = embedding_repr.last_hidden_state[1,1:6] # shape (5 x 1024)

# if you want to derive a single representation (per-protein embedding) for the whole protein
emb_0_per_protein = emb_0.mean(dim=0) # shape (1024)

print(f"emb_0.shape={emb_0.shape}, emb_1.shape={emb_1.shape}, emb_0_per_protein={emb_0_per_protein.shape}")
