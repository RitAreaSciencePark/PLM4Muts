from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import torch
import re 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained("Rostlab/ProstT5").to(device)

# only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
model.full() if device=='cpu' else model.half()

# prepare your protein sequences/structures as a list.
# Amino acid sequences are expected to be upper-case ("PRTEINO" below)
# while 3Di-sequences need to be lower-case.
sequence_examples = ["RSVASSKLWMLEFSAFLEQQQDPDTYNKHLFVHIGLEAVDIRQIYDKFPEKKGGLKDLFERGPSNAFFLVKFWADLNTNGSSFYGVSSQYESPENMIITCSTKVCSFGKQVVEKVETEYARYENGHYSYRIHRSPLCEYMINFIHKLKHLPEKYMMNSVLENFTILQVVTNRDTQETLLCIAYVFEVSASEHGAQHHIYRLVKE",
                     "RSVASSKLWMLEFSAFLEQQQDPDTYNKHLFVHIGLEAVDIRQIYDKFPEKKGGLKDLFERGPSNAFFLVKFWADLNTNGSSFYGVSSQYESPENMIITCSTKVCSFGKQVVEKVETEYARYENGHYSYRIHRSPLCEYMINFIHKLKHLPEKYMMNSVLENFTILQVVTNRDTQETLLCIAYVFEVSASEHGAQHHIHRLVKE",
                     "MPLDETNNESYRYLRSVGNTWKFNVEDVHPKMLERLYKRFDTFDLDTDGKMTMDEIMYWPDRMRQLVNATDEQVEKMRAAVHTFFFHKGVDPVNGLKREDWVEANRVFAEAERERERRGEPSLIALLSNAYYDVLDDDGDGTVDVEELKTMMKAFDVPQEAAYTFFQKADTDKTGKLERPELVHLFRKFWMEPYD",
                     "MPLDETNNESYRYLRSVGNTWKFNVEDVHPKMLERLYKKFDTFDLDTDGKMTMDEIMYWPDRMRQLVNATDEQVEKMRAAVHTFFFHKGVDPVNGLKREDWVEANRVFAEAERERERRGEPSLIALLSNAYYDVLDDDGDGTVDVEELKTMMKAFDVPQEAAYTFFQKADTDKTGKLERPELVHLFRKFWMEPYD",
                     "MPLDETNNESYRYLRSVGNTWKFNVEDVHPKMLERLYKRFDTFDLDTDGKMTMDEIMYWPDRMRQLVNATDEQVEKMRAAVHTFFFHKGVDPVNGLKREDWVEANRVFAEAERERERRGEPSLIALLSNAYYDVLDDDGDGTVDVEELKTMMKAFDVPQEAAYTFFQKADTDKTGKLERPELVHLFRKFWMEPYD",
                     "MPLDETNNESYRYLRSVGNTWKFNVEDVHPKMLERLYKRFDTFDLDTDGKMTMDEIMYWPDRMRQLVNATDEQVEKMRAAVHTFFFHKGVDPVNGLKREDWVEANRVFAEAERERERRGEPSLIALLGNAYYDVLDDDGDGTVDVEELKTMMKAFDVPQEAAYTFFQKADTDKTGKLERPELVHLFRKFWMEPYD",
                     "MPLDETNNESYRYLRSVGNTWKFNVEDVHPKMLERLYKRFDTFDLDTDGKMTMDEIMYWPDRMRQLVNATDEQVEKMRAAVHTFFFHKGVDPVNGLKREDWVEANRVFAEAERERERRGEPSLIALLSNAYYDVLDDDGDGTVDVEELKTMMKAFDVPQEAAYTFFQKADTDKTGKLERPELVHLFRKFWMEPYD",
                     "MPLDETNNESYRYLRSVGNTWKFNVEDVHPKMLERLYKRFDTFDLDTDGKMTMDEIMYWPDRMRQLVNATDEQVEKMRAAVHTFFFHKGVDPVNGLKREDWVEANRVFAEAERERERRGEPSLIALLSNAYYDVLDDDGDGTVDVEELKTMMKAFDVPQEAAYTFFQKADTDKTGKLERPELTHLFRKFWMEPYD",
        ]
min_len = min([ len(s) for s in sequence_examples])
max_len = max([ len(s) for s in sequence_examples])+10

# replace all rare/ambiguous amino acids by X (3Di sequences does not have those) and introduce white-space between all sequences (AAs and 3Di)
sequence_examples_p = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

# add pre-fixes accordingly. For the translation from AAs to 3Di, you need to prepend "<AA2fold>"
sequence_examples_p = [ "<AA2fold>" + " " + s for s in sequence_examples_p]

# tokenize sequences and pad up to the longest sequence in the batch
ids = tokenizer.batch_encode_plus(sequence_examples_p,
                                  add_special_tokens=True,
                                  padding="longest",
                                  return_tensors='pt').to(device)

# Generation configuration for "folding" (AA-->3Di)
gen_kwargs_aa2fold = {
                  "do_sample": True,
                  "num_beams": 3,
                  "top_p" : 0.95,
                  "temperature" : 1.2,
                  "top_k" : 6,
                  "repetition_penalty" : 1.2,
                  }

# translate from AA to 3Di (AA-->3Di)
with torch.no_grad():
  print(sequence_examples)
  print(ids.input_ids.shape, ids.attention_mask.shape)
  translations = model.generate( 
              ids.input_ids, 
              attention_mask=ids.attention_mask, 
              max_length=max_len, # max length of generated text
              min_length=min_len, # minimum length of the generated text
              early_stopping=True, # stop early if end-of-text token is generated
              num_return_sequences=1, # return only a single sequence
              **gen_kwargs_aa2fold
  )
# Decode and remove white-spaces between tokens
decoded_translations = tokenizer.batch_decode( translations, skip_special_tokens=True )
structure_sequences = [ "".join(ts.split(" ")) for ts in decoded_translations ] # predicted 3Di strings

print(sequence_examples)
print(structure_sequences)

if structure_sequences[0] != "dwlddpfktwpdkdkdkdfdddppdpdddddddddfaedapvvcqqlaapdvvipvvdcvvallsqekekekatelpddprmfifmktkmkgldffkkkkwkfkffsrdtpdididiwtfdddpnitmtmpprhgddpvvvvvsvvlspdpdqvvsqvrqvrikmwiwmagppprrtrhiyiygyghdpdpdhmdmgmhrhdhd":
    print("1 ERROR")
if structure_sequences[1] != "dwlddpfktwpdkdkdkdfdddppdpdddddddddfaedapvvcqqlaapdvvipvvdcvvallsqekekekatelpddprmfifmktkmkgldffkkkkwkfkffsrdtpdididiwgfdddpnitmtmpprhgddpvvvvvsvvlspdpdqvvsqvrqvrikmwiwmagppprrtrhiyiygyghdpdpdhmdmgmhrhdhd":
    print("2 ERROR")
if structure_sequences[2] != "ddddppppvvvvvvvvvppppppplvpddplllvlllvvqvvlppvppqfralvsllcvlvvlcvqqvddpvlsvllsvlsvvlcvvlvhdnvvghhsvsssvslsvllvvqvvcvvvvhdgsllsnlvsvqsslprvppqwhalvsqqsvcvsvvhdsvvsvvlqvvlppvpprthhsvsssvssccsrrnddd":
    print("3 ERROR")
if structure_sequences[3] != "ddddppppvvvvvvvvvvpppppplvpddplllvlllvvqvvlppvppqfhalvsllcvlvvlcvqqvddpvlsvllsvlsvvlcvvlvhdnvrghhsvsssvslsvllvvqvvcvvvvhdgsllsnlvsvqsslprvppqwhalvsqqsvcvsvvhdsvvsvvlqvvlppvpprthhsvsssvssccsrrnddd":
    print("4 ERROR")
if structure_sequences[4] != "ddddpppppvvvvvvvvvpppppplvpddpvvlvvllvvqvvlppvppqfralvsqlcvlvvlcvqqvddpvlsvlssvlsvllcvvlvhdnvvghhsvsssvslsvllvvqvvcvvvvhdhsllsnlvsvqsslprvppqwhalvsqqsvcvsvvhdsvvsvvlqvvlppvpprihhsvsssvssccvrrnddd":
    print("5 ERROR")
if structure_sequences[5] != "ddddppppvvvvvvvvvvpppppplvpddplllvlllvvqvvlppvppqfralvsllcvlvvlcvqqvddpvlsvllsvlsvvlcvvlvhdnvvghhsvsssvslsvllvvqvvcvvvvhdhsllsnlvsvqsslpsvppqwhalvsqqsvcvsvvhdsvvsvvlqvvlppvpprihhsvsssvssccsrrnddd":
    print("6 ERROR")
if structure_sequences[6] != "ddddpvvvvvvvvvvvvvppppddlvpddplllvlllvvqvvlppvppqfhalvsllvvlvvlcvqqvddpvlsvllsvlsvvlcvvlvhdnvvghhsvsssvslsvllvvqvvcvvvvhdhsllsnlvsvvsslprvppqwhalvsqqsvcssvvhdsvvsvvlqvvlppvpprthhsvsssvssccsrrnddd":
    print("7 ERROR")
if structure_sequences[7] != "ddddpppppvvvvvvvvvpppppplvpddplllvlllvvqvvlppvppqfhalvsqlcvlvvlcvqqvddpvlsvlssvlsvvlcvvlvhdnvvghhsvsssvslsvllvvqvvcvvvvhdhsllsnlvsvqsslprvppqwhalvsqvsvcvsvvhdsvvsvvlqvvlppvpprthhsvsssvssccsrhnddd":
    print("8 ERROR")
# Now we can use the same model and invert the translation logic
# to generate an amino acid sequence from the predicted 3Di-sequence (3Di-->AA)

# add pre-fixes accordingly. For the translation from 3Di to AA (3Di-->AA), you need to prepend "<fold2AA>"
sequence_examples_backtranslation = [ "<fold2AA>" + " " + s for s in decoded_translations]

# tokenize sequences and pad up to the longest sequence in the batch
ids_backtranslation = tokenizer.batch_encode_plus(sequence_examples_backtranslation,
                                  add_special_tokens=True,
                                  padding="longest",
                                  return_tensors='pt').to(device)

# Example generation configuration for "inverse folding" (3Di-->AA)
gen_kwargs_fold2AA = {
            "do_sample": True,
            "num_beams": 3,
            "top_p" : 0.90,
            "temperature" : 1.1,
            "top_k" : 6,
            "repetition_penalty" : 1.2,
}

# translate from 3Di to AA (3Di-->AA)
with torch.no_grad():
  backtranslations = model.generate( 
              ids_backtranslation.input_ids, 
              attention_mask=ids_backtranslation.attention_mask, 
              max_length=max_len, # max length of generated text
              min_length=min_len, # minimum length of the generated text
              early_stopping=True, # stop early if end-of-text token is generated
              num_return_sequences=1, # return only a single sequence
              **gen_kwargs_fold2AA
  )
# Decode and remove white-spaces between tokens
decoded_backtranslations = tokenizer.batch_decode( backtranslations, skip_special_tokens=True )
aminoAcid_sequences = [ "".join(ts.split(" ")) for ts in decoded_backtranslations ] # predicted amino acid strings

print(aminoAcid_sequences)


