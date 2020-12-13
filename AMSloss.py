import torch
import torch.nn as nn
import torch.nn.functional as F


class AdMSoftmaxLoss(nn.Module):

    def __init__(self, embedding_dim, no_classes, alpha = 1.0, margin=0.1):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.no_classes = no_classes
        self.embedding = nn.Embedding(no_classes, embedding_dim, max_norm=1)
        self.loss = nn.CrossEntropyLoss()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x, labels, m = None):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.no_classes

        n = len(labels)
        m = self.margin if m is None else m
        x = F.normalize(x, dim=1)
        w = self.embedding.weight        
        cos_theta = torch.matmul(w, x.T).T
        psi = cos_theta - m
        
        onehot = torch.zeros(n, self.no_classes)
        onehot[range(n), labels] = 1

        logits = self.alpha * torch.where(onehot == 1, psi, cos_theta)
        err = self.loss(logits, labels)
        idx = cos_theta.argmax(dim = 1)
        
        return err, logits, idx
