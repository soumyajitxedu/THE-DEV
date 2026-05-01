import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import urllib.request


print("using device is idk wait i found it still loading ")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"your using :{device} ")
#we have to download shakespare dataset from internet


# ════════════════════════════════
# 1. DOWNLOAD SHAKESPEARE
# ════════════════════════════════


url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
urllib.request.urlretrieve(url, 'shakespeare.txt')
with open('shakespeare.txt', 'r') as f:
    text = f.read()
print(f"Total characters :{len(text):,}")
print(f"sample: \n{text[:200]}")


# now we have convert into charater to function and learnthe model 
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════ 2. CHARACTER LEVEL TOKENIZER ══════════════════════════════════════════════════════════
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"vocab size : {vocab_size}")
print(f"characters : {''.join(chars)}")
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
print(encode("hello"))
print(decode(encode('hello')))
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════ 3. TRAIN/VAL SPLIT ═══════════════════════════════════════════════════════════════════
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"train : {len(train_data):,} val: {len(val_data):,}")

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════ 4. BATCH GENERATOR ═══════════════════════════════════════════════════════════════════
def get_batch(split,batch_size=32,block_size = 128):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)

#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════ 5.  MODEL  ═══════════════════════════════════════════════════════════════════
def scaled_dot_product_attention(Q,K,V , mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q,K.transpose(-2,-1))
    scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weigths = F.softmax(scores,dim=-1)
    return torch.matmul(weigths,V), weigths


# head 
class CausalMultiheadattention(nn.Module):
    def __init__(self,d_model,num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model// num_heads
        self.W_Q = nn.Linear(d_model,d_model)
        self.W_K = nn.Linear(d_model,d_model)
        self.W_V = nn.Linear(d_model,d_model)
        self.W_O = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
    def split_head(self,x,batch):
        x = x.view(batch,-1,self.num_heads,self.d_k)
        return x.transpose(1,2)
    def forward(self,x,mask=None):
        batch = x.size(0)
        Q = self.split_head(self.W_Q(x), batch)
        K = self.split_head(self.W_K(x), batch)
        V = self.split_head(self.W_V(x), batch)
        out, _ = scaled_dot_product_attention(Q,K,V,mask)
        out = out.transpose(1,2).contiguous()
        out = out.view(batch,-1,self.d_model)
        return self.W_O(out)




class feedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout = 0.1):