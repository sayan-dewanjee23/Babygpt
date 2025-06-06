# loading input text corpus
!wget https://raw.githubusercontent.com/sayan-dewanjee23/Babygpt/refs/heads/main/input.txt

# opening the file input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))

# building basic tokenizer
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

data = torch.tensor(encoder(text))    # creating tensor

# train-test split
train_data = data[:int(0.9*len(data))]
test_data = data[int(0.9*len(data)):]

# creating Bi-gram model
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import math
torch.manual_seed(1337)

# hyper parameters
vocab_size = 65
lrate = 1e-1                            # learning rate
block_size = 256                         # setting block size
batch_size = 64                          # setting batch size
max_iters = 5000
eval_iters = 50
patience = 7                             # how many evals to wait before stopping
best_val_loss = float('inf')
evals_since_improvement = 0

# creating function for getting batches
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
    return x_tensor,y_tensor

# bi-gram model
class BigramModel(nn.Module):

  def __init__(self,vocab_size):
    super().__init__()
    # embedding vectors over all vocabulary
    self.embedding_table = nn.Embedding(vocab_size,vocab_size)


  def forward(self,id,targets=None):
      # calculating logits and losses
      # id is a tensor of shape (B,T)

      logits = self.embedding_table(id)  # stacking logits based on those id hence shape is (B,T,vocab_size)

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

        logit,loss = self(id)                             # logit is obtained for new id with shape (B,T,C)
        logit = logit[:,-1,:]                             # considered the last id from every batch hence shape (B,C)
        probs = F.softmax(logit,dim=1)                    # turning logits into probability
        newid = torch.multinomial(probs,num_samples=1)    # sampling from multinomial distribution
        id = torch.cat((id,newid),dim=1)

      return id

model = BigramModel(vocab_size)

# initialising optimizer
optimizer  = torch.optim.AdamW(model.parameters(),lr = lrate)

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

# training loop
train_losses = []
val_losses = []

for iter in range(max_iters):
    # sample a batch of data
    xb, yb = build_batch('train')

    # forward pass
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # every eval_iters steps, check validation loss
    if iter % eval_iters == 0 or iter == max_iters - 1:
        train_loss = estimate_loss(model, data='train', batch_size=4, block_size=8)
        val_loss = estimate_loss(model, data='val', batch_size=4, block_size=8)  # use val here!
        
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

# creating new sequence
print(decoder(model.generator(id = torch.zeros((1, 1),dtype=torch.long), new_tokens=500)[0].tolist()))

# evaluating metric
def compute_val_perplexity(model, val_data, block_size=8):
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for i in range(0, len(val_data) - block_size):
            # x = input characters, y = next characters
            x = val_data[i:i+block_size].unsqueeze(0)  # (1, block_size)
            y = val_data[i+1:i+block_size+1].unsqueeze(0)  # (1, block_size)

            logits, loss = model(x, y)
            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / count
    perplexity = math.exp(avg_loss)

    return perplexity


val_perplexity = compute_val_perplexity(model, test_data, block_size=block_size)
print(f"Validation Perplexity: {val_perplexity:.2f}")