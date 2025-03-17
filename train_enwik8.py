import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from crossformers.crossformers import CrossFormers

# Konstanta
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
GENERATE_LENGTH = 1024
SEQ_LEN = 1024

# Helper functions
def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# Fungsi untuk generasi teks
def generate(model, prompts, seq_len, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device
    b, t = prompts.shape
    generated = prompts.clone()

    for i in range(seq_len):
        with torch.no_grad():
            # Gunakan slice akhir jika generated terlalu panjang untuk menghindari masalah memori
            input_seq = generated[:, -SEQ_LEN:] if generated.size(1) > SEQ_LEN else generated
            out = model(input_seq, input_seq)
            logits = out[:, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat((generated, next_token), dim=-1)

    return generated[:, t:]

# Persiapan data enwik8
def prepare_data():
    with gzip.open('./data/enwik8.gz') as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
        train_x, valid_x = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(train_x), torch.from_numpy(valid_x)
    return data_train, data_val

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq

    def __len__(self):
        return self.data.size(0) // self.seq_len

def main():
    # Instansiasi model CrossFormer
    dimensi_embedding = 256
    ukuran_vocab = 256  # Untuk enwik8 (ASCII)
    jumlah_block = 4
    heads = 8
    
    model = CrossFormers(
        dimensi_embedding=dimensi_embedding,
        ukuran_sumber_vocab=ukuran_vocab,
        ukuran_target_vocab=ukuran_vocab,
        panjang_sekuens=SEQ_LEN,
        jumlah_block=jumlah_block,
        faktor_ekspansi=4,
        heads=heads,
        dropout=0.1,
        norm_type="dyt"  # Menggunakan Dynamic Tanh normalization
    )
    
    # Persiapan data
    data_train, data_val = prepare_data()
    train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
    val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
    train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True))
    val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True))
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
        model.train()
        
        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            batch = next(train_loader)
            inp, target = batch[:, :-1], batch[:, 1:]
            output = model(inp, inp)  # menggunakan inp sebagai sumber dan target
            loss = F.cross_entropy(output.reshape(-1, ukuran_vocab), target.reshape(-1))
            (loss / GRADIENT_ACCUMULATE_EVERY).backward()
        
        print(f'training loss: {loss.item()}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        optimizer.zero_grad()
        
        if i % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                batch = next(val_loader)
                inp, target = batch[:, :-1], batch[:, 1:]
                output = model(inp, inp)
                loss = F.cross_entropy(output.reshape(-1, ukuran_vocab), target.reshape(-1))
                print(f'validation loss: {loss.item()}')
        
        if i % GENERATE_EVERY == 0:
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            prime = decode_tokens(inp)
            print(f'Prime text: \n{prime} \n\n {"*" * 100}')
            
            sample = generate(
                model=model,
                prompts=inp.unsqueeze(0),
                seq_len=GENERATE_LENGTH,
                temperature=0.8
            )
            
            output_str = decode_tokens(sample[0])
            print(f'Generated text: \n{output_str}')

if __name__ == "__main__":
    main()
