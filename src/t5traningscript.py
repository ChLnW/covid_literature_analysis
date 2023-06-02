################ TRAINING PARAMETERS START HERE #####################

maxInputLength = 1024 # max input length into the model, no hard limit from t5

regHead = 0.01 # regularization on header
regModel = 0.0 # regularization on pretrained model
lrHead = 5e-4 # header learning rate
lrModel = 5e-5 # pretrained model learning rate

# Configurations of AdamW
adamBetas = (.9, .98) 
adamEps = 1e-9

nEpoch = 10 # number of training epochs
batchsize = 32 
gradAccumulation = 4 
# nWarmupSteps = 2 * len(labeled) // batchsize
# lrSchedule = lambda step: 1 / np.sqrt(embDim) * np.min((1/np.sqrt(step), step/np.power(nWarmupSteps, 1.5)))
labelSmoothing = 0.1 # strength of label smoothing

nlayers = 1 # use the output hidden states of the last $nlayers layers

################ TRAINING PARAMETERS END HERE #####################

################ MODEL LOADING CONFIGURATION START HERE #####################

from os.path import exists
from os import makedirs
storagePath = '/g/data/il82/cw5285/'

# Path or Model Name for huggingface to load the model
modelLoadingPath = 'google/flan-t5-small'
# path to save checkpoints and evaluation results
modelSavingPath = storagePath + 'flan-t5-small/' 
if not exists(modelSavingPath):
   makedirs(modelSavingPath)
# name of training checkpoint
checkpointName = 'checkpoint'
# name of saved evaluation results
evaluationName = 'result'

# map1Path = storagePath + 'newLabelMappingModified.csv'
map2Path = storagePath + 'LabelInverseMapping.json'
# map3Path = storagePath + 'label_categorized_modified.csv'
abstractPath = storagePath + 'abstractLabeled.csv'

resumeEpoch = 9 # resume training from which epoch, 0 to start from scratch

################ MODEL LOADING CONFIGURATION END HERE #####################

import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)
import json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score
import transformers
transformers.logging.set_verbosity_error()
from accelerate import Accelerator
import random

from conditionalLoss import ConditionalLoss
from t5Clf import *


# with open('./LabelMapping.json', 'r') as f:
#    map1 = json.load(f)
with open(map2Path, 'r') as f:
   map2 = json.load(f)
# with open('./SubjectMapping.json', 'r') as f:
#    map3 = json.load(f)
labeled = pd.read_csv(abstractPath, na_filter=False)
labeled['label'] = labeled['label'].apply(eval)
# labeled.isnull().sum()
labels = list(map2.keys())
for ls in map2.values():
   labels += ls
labels.remove('Other')
mlb = MultiLabelBinarizer(classes=labels)
yEnc = torch.from_numpy(mlb.fit_transform(labeled['label'])).type(torch.float)

# get rid of dominating majority class
# cond = (yEnc * torch.from_numpy(1-mlb.transform([["Clinical and Health",'Medicine']]))).sum(-1)==0
# yEnc = yEnc[~cond]
# labeled = labeled[(~cond).numpy()]

categoryMasks = torch.from_numpy(mlb.transform([map2[k] for k in labels[:10]])).type(torch.float)

model = transformers.AutoModel.from_pretrained(modelLoadingPath)
tokenizer = transformers.AutoTokenizer.from_pretrained(modelLoadingPath)

toks = tokenizer("test", return_tensors="pt", max_length=maxInputLength, truncation=True, padding="longest")
outputs = model(**toks, decoder_input_ids=torch.zeros(1,1, dtype=(torch.int)), use_cache=False)
embDim = outputs.last_hidden_state.shape[-1]

def inputForT5(t:str, a:str):
    return f'infer academic subject. title: {t}. abstract: {a}'
Xtrain = [inputForT5(t,a) for t,a in zip(labeled['title'], labeled['abstract'])]
Xtrain = tokenizer(Xtrain, return_tensors="pt", max_length=maxInputLength, truncation=True, padding="longest")

# def f1Score(ypred, ytrue):
#    # ypred, ytrue: N x nClass
#    return np.array([f1_score(ytrue[:, i], ypred[:, i], zero_division=1) for i in range(ytrue.shape[-1])])

def recall(ypred, ytrue):
   # ypred, ytrue: N x nClass
   return np.array([recall_score(ytrue[:, i], ypred[:, i], zero_division=1) for i in range(ytrue.shape[-1])])

def precision(ypred, ytrue):
   # ypred, ytrue: N x nClass
   return np.array([precision_score(ytrue[:, i], ypred[:, i], zero_division=1) for i in range(ytrue.shape[-1])])

def evalModel(data:DataTransform, clf:t5Classifier, condLoss:ConditionalLoss, accelerator:Accelerator, returnMean=True, batchSize=64):
   clf.eval()
   data_loader = DataLoader(data, batch_size=batchSize, shuffle=False, collate_fn=collate)

   with torch.no_grad(): # turn of autograd to make everything faster
      all_out = [] # predictions made by the model
      all_y = [] # true classes (ground truth labels)
      for X, mask, decInput, y in data_loader:
         out = clf(X.to(accelerator.device), mask.to(accelerator.device), decInput.to(accelerator.device))
         # out = condLoss.genOutput(out)
         out, y = accelerator.gather_for_metrics((out, y.to(accelerator.device)))
         # store the predictions and ground truth labels
         all_out.append(out)
         all_y.append(y)
      # combine predictions, and true labels into two big vectors
      all_out = condLoss.genOutput(torch.cat(all_out, dim=0)).cpu().detach().numpy()
      all_y = torch.cat(all_y, dim=0).cpu().numpy()
      weights = all_y.sum(0)
      weights /= len(all_y)
      # f1s = f1Score(all_out, all_y)
      precisions = precision(all_out, all_y)
      recalls = recall(all_out, all_y)
      score = {'Precision':precisions, 'Recall':recalls, 'Weight':weights}
   clf.train()
   return score

# 10% used as valuation set
trainIdx, valIdx = train_test_split(range(len(labeled)), test_size=.1, shuffle=True, random_state=3820)

dataTrain = DataTransform(Xtrain, yEnc, trainIdx)
dataVal = DataTransform(Xtrain, yEnc, valIdx)

resumeEpoch = max(0, resumeEpoch)

def main():
   accelerator = Accelerator(gradient_accumulation_steps=gradAccumulation)

   clf = t5Classifier(model, embDim, categoryMasks.shape[-1], nlayers=nlayers).to(accelerator.device)
   condLoss = ConditionalLoss(categoryMasks, labelSmoothing).to(accelerator.device)

   opt = torch.optim.AdamW(clf.head.parameters(), lr=lrHead, weight_decay=regHead, betas=adamBetas, eps=adamEps)
   opt.add_param_group({'params':clf.t5.parameters(), 'lr':lrModel, 'weight_decay':regModel})
   # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', .7, 0, 5e-3, 'rel')

   clf, condLoss, opt = accelerator.prepare(clf, condLoss, opt)

   if resumeEpoch > 0:
      accelerator.load_state(modelSavingPath+ checkpointName + str(resumeEpoch))

   for epoch in range(resumeEpoch+1, resumeEpoch+nEpoch+1):
      sumLoss = 0.0
      # reshuffle every epoch
      data_loader = DataLoader(dataTrain, batch_size=batchsize, shuffle=True, collate_fn=collate)
      data_loader = accelerator.prepare_data_loader(data_loader, device_placement=True)
      loss = 0.0
      for X, mask, decInput, y in data_loader:
         with accelerator.accumulate(clf):
            opt.zero_grad()
            out = clf(X.to(accelerator.device), mask.to(accelerator.device), decInput.to(accelerator.device))
            loss = condLoss.lossConditional(out, y)
            sumLoss += float(loss)
            # loss.backward()
            accelerator.backward(loss)
            opt.step()
            
      # scheduler.step(sumLoss)

      train = evalModel(DataTransform(Xtrain, yEnc, # sample 15% of training data for evaluation
                                       random.sample(trainIdx, int(0.15*len(trainIdx)))
                                       ), clf, condLoss, accelerator)
      val = evalModel(dataVal, clf, condLoss, accelerator)
      dfTrain = pd.DataFrame(train, index=mlb.classes_)
      dfVal = pd.DataFrame(val, index=mlb.classes_)
      res = pd.concat({'Train':dfTrain, 'Val':dfVal}, axis=1)
      accelerator.print(f"epoch {epoch}: loss = {sumLoss}\n", flush=True)
      accelerator.print(res, '\n', flush=True)
      res.to_csv(modelSavingPath + evaluationName + str(epoch))
      # start saving model from epoch 4
      if epoch > 3:
         accelerator.save_state(modelSavingPath + checkpointName + str(epoch))

if __name__ == "__main__":
   main()