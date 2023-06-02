import torch

class DataTransform(torch.utils.data.Dataset):
   def __init__(self, X, y, idx):
      self.toks = X.input_ids[idx]
      self.masks = X.attention_mask[idx]
      self.y = y[idx]
      # self.decoderInput = torch.zeros(1, 1, dtype=torch.int).expand(len(X),1)
    
   def __len__(self):
      return len(self.y)
    
   def __getitem__(self, index):
      return self.toks[index], self.masks[index], self.y[index]

def collate(batch):
   x = torch.stack([v[0] for v in batch])
   mask = torch.stack([v[1] for v in batch])
   decInput = torch.zeros(len(x), 1, dtype=torch.int)
   y = torch.stack([v[2] for v in batch])
   return x, mask, decInput, y

class t5Head(torch.nn.Module):
   def __init__(self, embDim, numClass, nlayers=1):
      super().__init__()

      self.nlayers = nlayers
      self.W = torch.nn.Linear(nlayers*embDim, numClass, bias=True)

   def forward(self, hiddenStates):
      # hiddenStates: tuple(B*1*embDim)
      # out: B*(nlayers*embDim)
      out = torch.cat(hiddenStates[-self.nlayers:], dim=-1).squeeze(-2)
      # out: B x numClass
      out = self.W(out)
      return out
   
class t5Classifier(torch.nn.Module):
   def __init__(self, t5, embDim, numClass, nlayers=1):
      super().__init__()
      
      self.nlayers = nlayers
      self.t5 = t5
      self.head = t5Head(embDim, numClass, nlayers)
      

   def forward(self, X, mask, decoderInput):
      out = self.t5(input_ids=X, attention_mask=mask,
                  decoder_input_ids=decoderInput,
                  use_cache=False, output_hidden_states=True).decoder_hidden_states[-self.nlayers:]
      out = self.head(out)
      return out