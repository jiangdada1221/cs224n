#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import numpy as np

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder,self).__init__()
        self.charDecoder = nn.LSTM(input_size=char_embedding_size,hidden_size=hidden_size)
        self.char_output_projection = nn.Linear(hidden_size,len(target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(target_vocab.char2id),char_embedding_size,target_vocab.char2id['<pad>'])
        self.target_vocab = target_vocab
        self.char_loss = nn.CrossEntropyLoss(reduction='sum',ignore_index = target_vocab.char2id['<pad>'])
        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        
        chars_embedding = self.decoderCharEmb(input) #(length,batch,char_emb)        
        hs,dec_hidden = self.charDecoder(chars_embedding,dec_hidden)
        scores = self.char_output_projection(hs)
        return scores,dec_hidden
        
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        input_seq = char_sequence[:-1] #no <end> input sentence should not have the end, only need the start token
        scores, _ = self.forward(input_seq,dec_hidden) #(l,b,v_size)        
        
        target = char_sequence[1:].contiguous().view(-1)        
        scores= scores.view(scores.size()[0]*scores.size()[1],-1)        
        return self.char_loss(scores,target)
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        
        batch = initialStates[0].size()[1]
        outputs = [''] * batch
        reference = list(range(batch))
        inputs = torch.tensor([self.target_vocab.start_of_word]*batch,device=device).view((1,batch))
        dec_hidden = initialStates
        for t in range(max_length):
            scores, (h_,c_)= self.forward(inputs,dec_hidden) #scores : (l,b,v) (1,b,v)
            maxs = scores.argmax(dim=-1).squeeze(0).detach().numpy() #(b)               
            new_chars = [self.target_vocab.id2char[k] for k in maxs]            
            to_keep = [reference[k] for k in range(len(new_chars)) if new_chars[k] != '}'] #1,3,5,6...
            to_keep_hidden = [k for k in range(len(maxs)) if new_chars[k] != '}']
            # outputs = [outputs[k] + new_chars[i]  for i,k in enumerate(reference) if new_chars[i] != '}'] # errors
            for i,k in enumerate(reference):
                if new_chars[i] != '}':
                    outputs[k] = outputs[k] + new_chars[i]
            new_chars = [self.target_vocab.char2id[k] for k in new_chars if k != '}']
            reference=  to_keep 
            inputs = torch.tensor(new_chars,device=device).view(1,len(new_chars))            
            h_,c_ = h_[:,to_keep_hidden,:],c_[:,to_keep_hidden,:]
            dec_hidden = (h_,c_)
            if len(reference) == 0:
                break
        
        assert len(outputs) == batch    

        return outputs
        ### END YOUR CODE
