import math
import random
import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                          dim_feedforward, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
        return mask
    def forward(self, src, tgt):
        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_decoder(tgt_emb)
        src_emb = src_emb.transpose(0, 1)
        tgt_emb = tgt_emb.transpose(0, 1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb.size(0)).to(src.device)
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        output = self.fc_out(output)
        return output.transpose(0, 1)
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
def simple_tokenizer(text):
    return text.strip().split()

def build_vocab(sentences, min_freq=1):
    counts = Counter(token for sentence in sentences for token in sentence)
    vocab = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2}
    idx = len(vocab)
    for token, count in counts.items():
        if count >= min_freq and token not in vocab:
            vocab[token] = idx
            idx += 1
    return vocab
def numericalize(sentence, vocab):
    return [vocab[SOS_TOKEN]] + [vocab[token] for token in sentence if token in vocab] + [vocab[EOS_TOKEN]]
class PrepareDataset(Dataset):
    def __init__(self, data, src_vocab=None, tgt_vocab=None, build_vocabs=False):
        self.df = data.copy()
        
        # Fill missing values in both columns
        self.df["text"] = self.df["text"].fillna("")
        self.df["code"] = self.df["code"].fillna("")
        # Tokenize the source and target strings
        self.df["src_tokens"] = self.df["code"].apply(simple_tokenizer)
        self.df["tgt_tokens"] = self.df["text"].apply(simple_tokenizer)

        if build_vocabs:
            self.src_vocab = build_vocab(self.df["src_tokens"].tolist())
            self.tgt_vocab = build_vocab(self.df["tgt_tokens"].tolist())
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        self.df["src_indices"] = self.df["src_tokens"].apply(lambda tokens: numericalize(tokens, self.src_vocab))
        self.df["tgt_indices"] = self.df["tgt_tokens"].apply(lambda tokens: numericalize(tokens, self.tgt_vocab))
        self.data = list(zip(self.df["src_indices"].tolist(), self.df["tgt_indices"].tolist()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_tensors = [torch.tensor(seq, dtype=torch.long) for seq in src_batch]
    tgt_tensors = [torch.tensor(seq, dtype=torch.long) for seq in tgt_batch]
    src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=0)
    return src_padded, tgt_padded
dft = pd.read_csv("spoc-train-train.tsv", sep="\t")
dfe = pd.read_csv("spoc-train-eval.tsv", sep="\t")
dfts = pd.read_csv("spoc-train-test.tsv", sep="\t")

first_two_columns_train = dft.iloc[:, :2]
first_two_columns_eval = dfe.iloc[:, :2]
first_two_columns_test = dfts.iloc[:, :2]

train_dataset = PrepareDataset(first_two_columns_train, build_vocabs=True)
eval_dataset = PrepareDataset(first_two_columns_eval, src_vocab=train_dataset.src_vocab,
                                 tgt_vocab=train_dataset.tgt_vocab, build_vocabs=False)
test_dataset = PrepareDataset(first_two_columns_test, src_vocab=train_dataset.src_vocab,
                                 tgt_vocab=train_dataset.tgt_vocab, build_vocabs=False)

# Load model
def generate_output(model, src_sentence, src_vocab, tgt_vocab, device, max_len=50):
    model.eval()
    tokens = simple_tokenizer(src_sentence)
    src_indices = numericalize(tokens, src_vocab)
    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)
    tgt_indices = [tgt_vocab[SOS_TOKEN]]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
        next_token = torch.argmax(output[0, -1, :]).item()
        tgt_indices.append(next_token)
        if next_token == tgt_vocab[EOS_TOKEN]:
            break
    inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
    generated_tokens = [inv_tgt_vocab[idx] for idx in tgt_indices if idx not in (tgt_vocab[SOS_TOKEN], tgt_vocab[EOS_TOKEN])]
    return " ".join(generated_tokens)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(
    src_vocab_size=len(train_dataset.src_vocab),
    tgt_vocab_size=len(train_dataset.tgt_vocab)
).to(device)

model.load_state_dict(torch.load("transformer_code.pth", map_location=device))
model.eval()


# Define inference function
def generate_pseudocode(code):
    generated_pseudo = generate_output(model, code, train_dataset.src_vocab, train_dataset.tgt_vocab, device)
    return generated_pseudo

# Gradio UI
demo = gr.Interface(
    fn=generate_pseudocode,
    inputs=gr.Textbox(lines=5, placeholder="Enter C++ code here..."),
    outputs=gr.Textbox(label="Generated Pseudocode"),
    title="Code to Pseudocode Generator",
    description="Enter C++ code, and the model will generate pseudocode."
)

demo.launch()
