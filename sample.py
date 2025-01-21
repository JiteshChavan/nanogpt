import os
import torch
from gpt_class import GPTConfig
from gpt_class import GPT
import torch.nn.functional as F
import tiktoken

log_dir = "log"
checkpoint_file_name = "model_95364.pt"
checkpoint_file = os.path.join(log_dir, checkpoint_file_name)

checkpoint = torch.load (checkpoint_file)



torch.manual_seed(1337)
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.manual_seed(1337)
else:
    device = 'cpu'
device = 'cpu'
model = GPT (GPTConfig (vocab_size=50304))
model.to(device)
model.load_state_dict (checkpoint['model'])

model.eval()
num_return_sequences = 10
max_length = 50

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello I'm a language model,")
tokens = torch.tensor (tokens, dtype=torch.long)

tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
xgen = tokens.to(device)
# Separate generator object for sampling, because I don't want to impact the rng state of the random number generator that is the global one
# used for training. I want this to be completely outside the training loop.
sample_rng = torch.Generator(device=device)
sample_rng.manual_seed (1337)
while xgen.size(1) < max_length:
    
    # forward the model and get the logits
    with torch.no_grad ():
        logits, loss = model (xgen) # (B, T, Vocab_size)
        # take the logits at the last position
        logits = logits [:,-1,:] # (B, Vocab_size)
        # get the probabilities
        probs = F.softmax (logits, dim=-1) # (B, Vocab_size)
        # do top-k sampling of 50
        # top-k probs here becomes (4, 50) top-k indices is becomes (4,50)
        topk_probs, topk_indices = torch.topk (probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: Multinomial does not demand the input to sum to 1
        ix = torch.multinomial (topk_probs, 1, generator=sample_rng) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather (topk_indices, -1, ix)
        # append to the sequence
        xgen= torch.cat ((xgen,xcol), dim=1)
# print the generated text
for i in range (num_return_sequences):
    tokens = xgen[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print (f"sample {i} : {decoded}")