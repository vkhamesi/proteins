from torch.utils.data import Dataset
from transformers import AutoTokenizer

class ProteinsDataset(Dataset):

    def __init__(self, protein_to_seq, go_to_desc, functions):

        self.protein_list = list(protein_to_seq.keys())
        self.protein_to_seq = protein_to_seq
        self.go_list = list(go_to_desc.keys())
        self.go_to_desc = go_to_desc
        self.functions = functions

        self.protein_tokenizer = AutoTokenizer.from_pretrained("yarongef/DistilProtBert")
        self.go_tokenizer = AutoTokenizer.from_pretrained("nlpie/distil-biobert")

    def __len__(self):
        return len(self.protein_to_seq) * len(self.go_to_desc)

    def __getitem__(self, idx):

        # Retrieve the protein sequence
        protein_idx = idx // len(self.go_to_desc)
        protein_name = self.protein_list[protein_idx]
        protein_sequence = " ".join(str(self.protein_to_seq[protein_name].seq))

        # Retrieve the gene ontology description
        go_idx = idx - protein_idx * len(self.go_to_desc)
        go_name = self.go_list[go_idx]
        go_description = self.go_to_desc[go_name]

        # Retrieve the label
        label = float(go_name in self.functions[protein_name])

        # Tokenize sequences
        protein_inputs = self.protein_tokenizer.encode_plus(protein_sequence, padding="max_length",
                                                            max_length=1024, truncation=True, return_tensors="pt")
        go_inputs = self.go_tokenizer.encode_plus(go_description, padding="max_length",
                                                  max_length=256, truncation=True, return_tensors="pt")

        protein_input_ids = protein_inputs["input_ids"].squeeze()
        protein_attention_mask = protein_inputs["attention_mask"].squeeze()
        go_input_ids = go_inputs["input_ids"].squeeze()
        go_attention_mask = go_inputs["attention_mask"].squeeze()

        return protein_input_ids, protein_attention_mask, go_input_ids, go_attention_mask, label