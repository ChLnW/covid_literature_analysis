import torch

class ConditionalLoss(torch.nn.Module):
   def __init__(self, categoryMasks, labelSmoothing=0.0):
      super().__init__()
      self.categoryMasks = torch.nn.parameter.Parameter(categoryMasks, requires_grad=False)
      self.labelSmoothing = labelSmoothing

   def smoothLabel(y, labelSmoothing=0.0):
      return labelSmoothing + (1-2*labelSmoothing) * y

   def multiLabelLoss(pred, true, mask):
      loss = true * torch.nn.functional.logsigmoid(pred) + (1-true) * torch.nn.functional.logsigmoid(-pred)
      loss *= (-mask)
      return loss.sum(-1).mean()

   def lossConditional(self, ypred, ytrue, labelSmoothing=None):
      labelSmoothing = self.labelSmoothing if labelSmoothing is None else labelSmoothing
      # ypred, ytrue: N x nClass
      nPred = len(self.categoryMasks)
      precondTrue = ytrue[:, :nPred]
      # mask: N x condClass
      mask = torch.matmul(precondTrue, self.categoryMasks)
      mask[:, :nPred] = 1.0
      return ConditionalLoss.multiLabelLoss(ypred, ConditionalLoss.smoothLabel(ytrue, labelSmoothing), mask)
   
   def genOutput(self, ypred):
      nPred = len(self.categoryMasks)
      precond = torch.round(torch.sigmoid(ypred))
      mask = torch.matmul(precond[..., :nPred], self.categoryMasks)
      res = precond * mask
      res[..., :nPred] += precond[..., :nPred]
      return res

   def forward(self, ypred, ytrue):
      # returns conditional loss
      return self.lossConditional(ypred, ytrue)