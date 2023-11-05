from src.dataset import ProteinsDataset
from src.model import ProteinPredictor
import argparse
import numpy as np
import pandas as pd
import obonet
from Bio import SeqIO
from tqdm import trange, tqdm
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

def extract_data(max_proteins, max_functions):
    
    np.random.seed(42)
    
    # Mapping between protein sequences and functions
    mlb = MultiLabelBinarizer()
    train_terms = pd.read_csv("./data/train_terms.tsv", sep="\t")
    functions = train_terms.groupby("EntryID")["term"].apply(list)
    functions_df = pd.DataFrame(mlb.fit_transform(functions), columns=mlb.classes_, index=functions.index)
    functions_df = functions_df[functions_df.sum(axis=0).sort_values(ascending=False)[:max_functions].index]
    functions_df = functions_df.loc[functions_df.sum(axis=1).sort_values(ascending=False)[:max_proteins].index]
    functions_df = functions_df[np.random.permutation(functions_df.columns)].sample(frac=1)
    functions = functions_df.apply(lambda row: row.index[row == 1].tolist(), axis=1).to_dict()

    # Text descriptions of functions
    go_graph = obonet.read_obo("./data/go-basic.obo")
    go_to_desc = {id_: (data.get("namespace") + data.get("def")).replace("_", " ").replace('"', " ") for id_, data in go_graph.nodes(data=True)}
    go_to_desc = {k: go_to_desc[k] for k in functions_df.columns}
    
    # Amino acids sequences of proteins
    protein_to_seq = SeqIO.to_dict(SeqIO.parse(open("./data/train_sequences.fasta"), "fasta"))
    protein_to_seq = {k: protein_to_seq[k] for k in functions_df.index}

    return protein_to_seq, go_to_desc, functions_df, functions

def truncate_dict(d, n):
    
    return {k: d[k] for k in list(d)[:n]}


def build_data(protein_to_seq, go_to_desc, functions, batch_size):
    
    dataset = ProteinsDataset(protein_to_seq, go_to_desc, functions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataset, dataloader


def build_peft_model(lora_rank, lora_dropout, target_modules, modules_to_save, device):
    
    config = LoraConfig(
        r=lora_rank,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save
    )
    
    model = ProteinPredictor()
    peft_model = get_peft_model(model, config)
    model = peft_model.to(device)
    
    return model

def pos_weight(functions_df, n_prot, n_func):
    
    weight = functions_df.iloc[:n_prot][functions_df.columns[:n_func]].mean().mean()
    
    return (1 - weight) / weight


def train_model(n_prot, n_func, batch_size, lora_rank, lora_dropout, target_modules, modules_to_save, device, lr, epochs, name):
    
    print("Building training dataset and dataloader.")
    protein_to_seq, go_to_desc, functions_df, functions = extract_data(1000, 1000)
    protein_to_seq = truncate_dict(protein_to_seq, n_prot)
    go_to_desc = truncate_dict(protein_to_seq, n_func)
    dataset, dataloader = build_data(protein_to_seq, go_to_desc, functions, batch_size)
    print("")
    
    print("Building model.")
    model = build_peft_model(lora_rank, lora_dropout, target_modules, modules_to_save, device)
    
    weight = pos_weight(functions_df, n_prot, n_func)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight).to(device))
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    bar = trange(epochs)
    loss_history = []

    for _ in range(epochs):

        model.train()
        for protein_input_ids, protein_attention_mask, go_input_ids, go_attention_mask, labels in tqdm(dataloader):

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):

                protein_input_ids = protein_input_ids.to(device)
                protein_attention_mask = protein_attention_mask.to(device)
                go_input_ids = go_input_ids.to(device)
                go_attention_mask = go_attention_mask.to(device)
                labels = labels.to(device)

                outputs = model(protein_input_ids, protein_attention_mask, go_input_ids, go_attention_mask)
                loss = criterion(outputs, labels.unsqueeze(1).float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            loss_history.append(loss.item())
            bar.set_postfix(loss=f"{np.mean(loss_history[-dataset.__len__()//batch_size:]):,.3f}")
    print("")
    
    print("Saving model.")
    torch.save(model, f"./models/{name}.pt")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Training the model.")
    parser.add_argument("--n_prot", type=int, help="Number of proteins.")
    parser.add_argument("--n_func", type=int, help="Number of functions.")
    parser.add_argument("--batch_size", type=int, help="Batch size.")
    parser.add_argument("--lora_rank", type=int, help="Rank of LoRA decomposition.")
    parser.add_argument("--lora_dropout", type=float, help="Dropout for LoRA layers.")
    parser.add_argument("--device", type=str, help="Device to be used for training, either 'cpu' or 'gpu'.")
    parser.add_argument("--lr", type=float, help="Learning rate for the AdamW optimiser.") 
    parser.add_argument("--epochs", type=int, help="Number of epochs.") 
    parser.add_argument("--name", type=str, help="Filename for the trained PyTorch model.") 
    
    target_modules=[
        "go_model.encoder.layer.0.output.dense",
        "go_model.encoder.layer.3.output.dense",
        "go_model.encoder.layer.5.output.dense",
        "protein_model.encoder.layer.13.output.dense",
        "protein_model.encoder.layer.14.output.dense"]
    
    modules_to_save=[
        "protein_reshape",
        "go_reshape",
        "mlp.0",
        "mlp.2"]
    
    args = parser.parse_args()
    
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_model(n_prot=args.n_prot, n_func=args.n_func, batch_size=args.batch_size, lora_rank=args.lora_rank, 
                lora_dropout=args.lora_dropout, target_modules=target_modules, modules_to_save=modules_to_save, 
                device=device, lr=args.lr, epochs=args.epochs, name=args.name)