# loading dataset
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# opening the file input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))

# unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# a final model implementing transformer blocks and dropout
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import math

# hyperparameter
batch_size = 64
block_size = 256
num_head = 6
head_size = 64
embed_length = num_head*head_size
n_layer = 6
dropout = 0.2
eval_iters = 50
max_iters = 5000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
patience = 7                             # how many evals to wait before stopping
best_val_loss = float('inf')
evals_since_improvement = 0

# building encoder-decoder
def encoder(input):
    output_list = []

    for char in input:
      for i in range(65):
        if char == chars[i]:
          output_list.append(i)

    return output_list

def decoder(input):
    output = ""

    for num in list(input):
      output += chars[num]

    return output

# transforming into tensor
data = torch.tensor(encoder(text)) 

# train-test split with first 90% data as training data
train_data = data[:int(0.9*len(data))]
test_data = data[int(0.9*len(data)):]

# building data with block-batch split
def build_batch(input):

    data = train_data if input == "train" else test_data
    a = torch.randint(0,len(data)-block_size,(batch_size,))
    x = []
    y = []

    for i in a:
      x.append(data[i : i+block_size].unsqueeze(0))
      y.append(data[i+1 : i+block_size+1].unsqueeze(0))

    x_tensor = torch.cat(x, dim=0)
    y_tensor = torch.cat(y, dim=0)
    x_tensor,y_tensor = x_tensor.to(device), y_tensor.to(device)
    return x_tensor,y_tensor


# single-head self attention
class self_attention(nn.Module):

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(embed_length ,head_size,bias = True)
    self.query = nn.Linear(embed_length ,head_size,bias = True)
    self.value = nn.Linear(embed_length ,head_size,bias = True)
    self.dropout = nn.Dropout(dropout)


  def forward(self,x):
      B,T,C = x.shape
      k = self.key(x)
      q = self.query(x)
      v = self.value(x)                           # shape is (B,T,head_size)
      weight = torch.bmm(q,k.transpose(1,2))* (k.shape[-1]**-0.5)      # result is (B,T,T)
      mask = torch.tril(torch.ones(T, T)).bool()  # shape (T, T)
      mask = mask.to(weight.device)
      weight = weight.masked_fill(~mask, float('-inf'))  # ~mask == upper triangle
      weight = F.softmax(weight,dim = -1)
      weight = self.dropout(weight)
      output = weight @ v                         # result is (B,T,head_size)

      return output


# multihead self-attention
class multihead(nn.Module):
  def __init__(self,num_head,head_size):
    super().__init__()
    self.heads = nn.ModuleList([self_attention(head_size) for _ in range(num_head)])
    self.layer_1 = nn.Linear(head_size*num_head,embed_length)
    self.drop_out = nn.Dropout(dropout)

  def forward(self,x):
    output = []
    for h in self.heads:
        output.append(h(x))
    output = torch.cat(output, dim=-1)
    output = self.drop_out(self.layer_1(output))

    return output

# creating feedforward class
class feedforward(nn.Module):
  """ a simple linear layer followed by a non-linearity """

  def __init__(self,input_dim):
    super().__init__()
    self.l1 = nn.Sequential(
        nn.Linear(input_dim,4*input_dim),
        nn.ReLU(),
        nn.Linear(4*input_dim,input_dim),
        nn.Dropout(dropout)
    )

  """ Here we have used output dimension of every layer as 4 times of input dimension.
   In above mentioned paper input dimension was 512 and output was 2048. Hence we have
   used that here """

  def forward(self,x):
      return self.l1(x)


# creating transformer block
class transformer(nn.Module):
  """ A single transformer block """

  def __init__(self,num_head,head_size):
    super().__init__()
    embed_length = num_head*head_size
    self.multi_head = multihead(num_head,head_size)
    self.lay_norm_1 = nn.LayerNorm(embed_length)
    self.lay_norm_2 = nn.LayerNorm(embed_length)
    self.ffwd = feedforward(input_dim = embed_length)

  """ Here we have used layer norm before processing through multihead attention.
     In paper multihead attention was worked out before that. And we have added
     input x in both layers with corresponding outputs. This is residual connection """

  def forward(self,x):
    x = x + self.multi_head(self.lay_norm_1(x))
    x = x + self.ffwd(self.lay_norm_2(x))

    return x


# creating final gpt model with transfer block
class final_GPTModel(nn.Module):

  def __init__(self,vocab_size):
    super().__init__()
    # embedding vectors over all vocabulary
    self.token_embedding = nn.Embedding(vocab_size,embed_length)
    self.position_embedding = nn.Embedding(block_size,embed_length)      # block_size is maximum sequence length in a batch
    self.sa_multihead = multihead(num_head,head_size)
    self.blocks = nn.Sequential(*[transformer(num_head,head_size) for _ in range(n_layer)],
                                nn.LayerNorm(embed_length))
    self.feed_forw_1 = nn.Linear(embed_length,vocab_size)
    self.final_norm = nn.LayerNorm(embed_length)

  def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
          if module.bias is not None:
              torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)        # weight initialization

  def forward(self,id,targets=None):
      # calculating logits and losses
      # id is a tensor of shape (B,T)

      B,T = id.shape                                                  # batch size * block size

      tk = self.token_embedding(id)                                   # B * T * embed_dim
      pos = self.position_embedding(torch.arange(T,device=device))                  # T * embed_dim
      x = tk + pos                                                    # broadcasting over two tensors
      x = self.blocks(x)                                              # B * T * embed_dim
      x = self.final_norm(x)
      logits = self.feed_forw_1(x)                                    # B * T * vocab_size


      if targets is None :
        losses = None
      else:
        B,T,C = logits.shape
        logits = logits.view(B*T,C)
        targets = targets.view(B*T)
        losses = F.cross_entropy(logits,targets)

      return logits,losses

  def generator(self,id,new_tokens):

      # here we are starting with a new id with dimension B*T

      for a in range(new_tokens):

        id_cond = id[:, -block_size:]

        logit,loss = self(id_cond)                             # logit is obtained for new id with shape (B,T,C)
        logit = logit[:,-1,:]                             # considered the last id from every batch hence shape (B,C)
        probs = F.softmax(logit,dim=1)                    # turning logits into probability
        newid = torch.multinomial(probs,num_samples=1)    # sampling from multinomial distribution
        id = torch.cat((id,newid),dim=1)

      return id
  

# calculating loss
@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, eval_iters=100):
    model.eval()
    losses = []

    for _ in range(eval_iters):

        xb, yb = build_batch(data)
        logits, loss = model(xb, yb)  # assumes model returns (logits, loss)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


torch.manual_seed(1337)
model = final_GPTModel(vocab_size = 65)
print(device)
model = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters()), 'parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train_losses = []
val_losses = []

for iter in range(max_iters):

    # sample a batch of data
    xb, yb = build_batch('train')
    xb = xb.to(device)
    yb = yb.to(device)
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_iters == 0 or iter == max_iters - 1:


        train_loss = estimate_loss(model,data = 'train',batch_size=batch_size,block_size=block_size)
        val_loss = estimate_loss(model,data = 'val',batch_size=batch_size,block_size=block_size)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            evals_since_improvement = 0
        else:
            evals_since_improvement += 1
            print(f" No improvement for {evals_since_improvement} evaluations.")

            if evals_since_improvement >= patience:
                print(" Early stopping triggered.")
                break



# Create x-axis: the iteration steps where loss was recorded
steps = list(range(0, len(train_losses) * eval_iters, eval_iters))

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(steps, train_losses, label='Train Loss')
plt.plot(steps, val_losses, label='Validation Loss')
plt.xlabel('Training Iterations')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# evaluation metric
val_loss = estimate_loss(model, data='test', block_size=block_size, batch_size=64, eval_iters=100)
val_perplexity = math.exp(val_loss)
print(f"Validation Perplexity: {val_perplexity:.2f}")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decoder(model.generator(context, new_tokens=2000)[0].tolist()))