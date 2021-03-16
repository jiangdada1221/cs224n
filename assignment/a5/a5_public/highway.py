#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self,e_word,dropout=0):
        super(Highway, self).__init__()
        self.Wproj = nn.Linear(e_word,e_word)        
        self.Wgate = nn.Linear(e_word,e_word)
        self.dropout = nn.Dropout(dropout)
    def forward(self,X_conv):
        '''
        @param X_conv : should be batch first, in size b x e_word
        '''
        X_proj = self.Wproj(X_conv)
        X_proj = F.relu(X_proj)
        X_gate = torch.sigmoid(self.Wgate(X_conv))
        X_highway = X_proj * X_gate + (1-X_gate) * X_conv 
        X_highway = self.dropout(X_highway)
        return X_highway


