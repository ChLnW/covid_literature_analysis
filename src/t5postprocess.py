import pandas as pd
import json
from sklearn.preprocessing import MultiLabelBinarizer
import transformers
import torch
from torch.utils.data import DataLoader
from conditionalLoss import ConditionalLoss
from t5Clf import *
from accelerate import Accelerator

storagePath = '/g/data/il82/cw5285/'
# storagePath = './'

# Path or Model Name for huggingface to load the model
modelLoadingPath = 'google/flan-t5-small'
# path to save checkpoints and evaluation results
modelSavingPath = storagePath + 'flan-t5-small/' 
checkpointName = 'checkpoint'

map2Path = storagePath + 'LabelInverseMapping.json'
abstractPath = storagePath + 'parsedAbstract.csv'

resumeEpoch = 14

maxInputLength = 1024

nlayers = 1

data = pd.read_csv(abstractPath)
data.date = data.date.apply(lambda x: pd.to_datetime(x).date())
isnull = data.label.isnull()
# labeled = data[~isnull]
unlabeled = data[isnull]
# labeled.label = labeled.label.apply(eval)

with open(map2Path, 'r') as f:
   map2 = json.load(f)
labels = list(map2.keys())
for ls in map2.values():
   labels += ls
labels.remove('Other')
mlb = MultiLabelBinarizer(classes=labels)

model = transformers.AutoModel.from_pretrained(modelLoadingPath)
tokenizer = transformers.AutoTokenizer.from_pretrained(modelLoadingPath)

toks = tokenizer("test", return_tensors="pt", max_length=maxInputLength, truncation=True, padding="longest")
outputs = model(**toks, decoder_input_ids=torch.zeros(1,1, dtype=(torch.int)), use_cache=False)
embDim = outputs.last_hidden_state.shape[-1]

def inputForT5(t:str, a:str):
    return f'infer academic subject. title: {t}. abstract: {a}'
Xtrain = [inputForT5(t,a) for t,a in zip(unlabeled['title'], unlabeled['abstract'])]

def evalModel(data, clf:t5Classifier, condLoss:ConditionalLoss, accelerator:Accelerator, batchSize=64):
   clf.eval()
   data_loader = DataLoader(data, batch_size=batchSize, shuffle=False)

   with torch.no_grad(): # turn of autograd to make everything faster
      all_out = [] # predictions made by the model
      for t in data_loader:
         toks = tokenizer(t, return_tensors="pt", max_length=maxInputLength, truncation=True, padding="longest")
         X = toks.input_ids
         mask = toks.attention_mask
         decInput = torch.zeros(len(X), 1, dtype=torch.int)
         out = clf(X.to(accelerator.device), mask.to(accelerator.device), decInput.to(accelerator.device))
         # out = condLoss.genOutput(out)
         out = accelerator.gather_for_metrics((out))
         # store the predictions and ground truth labels
         all_out.append(out)
      # combine predictions, and true labels into two big vectors
      all_out = condLoss.genOutput(torch.cat(all_out, dim=0)).cpu().detach()
      torch.save(all_out, modelSavingPath+'unlabelOut')
   return

gradAccumulation = 4 
labelSmoothing = 0.1

categoryMasks = torch.from_numpy(mlb.fit_transform([map2[k] for k in labels[:10]])).type(torch.float)

def main():
   accelerator = Accelerator(gradient_accumulation_steps=gradAccumulation)

   clf = t5Classifier(model, embDim, categoryMasks.shape[-1], nlayers=nlayers).to(accelerator.device)
   condLoss = ConditionalLoss(categoryMasks, labelSmoothing).to(accelerator.device)

   clf, condLoss = accelerator.prepare(clf, condLoss)

   accelerator.load_state(modelSavingPath+ checkpointName + str(resumeEpoch))

   evalModel(Xtrain, clf, condLoss, accelerator)

if __name__ == "__main__":
   main()