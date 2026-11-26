"""A small, model-free cross-attention example. Only PyTorch required.

Run: python cross_attention_toy.py
"""
import torch
import math

# Three 2-dimensional audio frame keys. Their values can be different in real models.
K_audio = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])  # [3,2]
V_audio = torch.tensor([[2., 0.], [0., 4.], [1., 1.]])  # [3,2]
Q_text = torch.tensor([[1., 0.], [0., 1.]])             # [2,2]

scores = Q_text @ K_audio.T / math.sqrt(K_audio.shape[1])  # [2,3]
weights = torch.softmax(scores, dim=-1)
context = weights @ V_audio  # [2,2]

print('Q_text:', Q_text.shape, '\n', Q_text)
print('K_audio:', K_audio.shape, '\n', K_audio)
print('V_audio:', V_audio.shape, '\n', V_audio)
print('attention scores:', scores.shape, '\n', scores)
print('attention weights:', weights.shape, '\n', weights)
print('retrieved audio context:', context.shape, '\n', context)
print('row sums:', weights.sum(-1))
