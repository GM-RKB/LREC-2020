from collections import *
from random import random
import pprint
import operator
import numpy as np
import json


class LanguageModel(object):

  def __init__(self, order=4, wi=1, add_k=0, lm={}, rightToLeft=False):
    '''order: The length of the n-grams.
    '''
    self.order = order
    self.add_k = add_k
    self.lm = lm
    self.wi = wi
    self.rightToLeft = rightToLeft

  def rev(self, s): return s[::-1]

  def train_char_lm(self, text_data,ignoreNum=True):
    ''' Trains a language model.

    This code was borrowed from 
    http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139

    Inputs:
      fname: Path to a text corpus.
      add_k: k value for add-k smoothing.

    Returns:
      A dictionary mapping from n-grams of length n to a list of tuples.
      Each tuple consists of a possible net character and its probability.
    '''
    if self.rightToLeft:
      text_data = self.rev(text_data)

    add_k = self.add_k
    order = self.order
    wi = self.wi
    lm = defaultdict(Counter)
    lm_cn = defaultdict(int)
    pad = "~" * order
    text_data = pad + text_data
    for i in range(len(text_data)-order):
      history, char = text_data[i:i+order], text_data[i+order:i+order+wi]
      if not ignoreNum or (not("".join(char).isnumeric()) and not("".join(history).isnumeric())):
        lm[history][char] += 1
        lm_cn[history] += 1

    def normalize(counter):
      s = float(sum(counter.values()))+float(len(counter)*add_k)
      return [(c, (cnt+add_k)/s) for c, cnt in counter.items()]
    outlm = {hist: normalize(chars) for hist, chars in lm.items()}
    self.lm = outlm
    self.lm_cn = lm_cn
    return outlm, lm_cn

  def print_probs(self, history, c=None):
    try:
      probs = dict(self.lm[history])
      if c:
        try:
          return probs[c]
        except:
          return 0
      else:
        return probs
    except:
        return 0

  def generate_letter(self, history):
    order = self.order
    lm = self.lm
    ''' Randomly chooses the next letter using the language model.
    
    Inputs:
      lm: The output from calling train_char_lm.
      history: A sequence of text at least 'order' long.
      order: The length of the n-grams in the language model.
      
    Returns: 
      A letter
    '''

    history = history[-order:]
    dist = lm[history]
    x = random()
    for c, v in dist:
      x = x - v
      if x <= 0:
        return c

  def generate_text(self, nletters=500):
    lm = self.lm
    '''Generates a bunch of random text based on the language model.
    
    Inputs:
    lm: The output from calling train_char_lm.
    history: A sequence of previous text.
    order: The length of the n-grams in the language model.
    
    Returns: 
      A letter  
    '''
    order = self.order
    history = "~" * order
    out = []
    for i in range(nletters):
      c = self.generate_letter(history)
      history = history[-order:] + c
      out.append(c)
    return "".join(out)

  def perplexity(self, text):
    lm = self.lm
    order = self.order
    '''Computes the perplexity of a text file given the language model.
    
    Inputs:
      test_filename: path to text file
      lm: The output from calling train_char_lm.
      order: The length of the n-grams in the language model.
    '''
    perp = 10
    N = 0

    test = text
    pad = "~" * order
    test = pad + test
    for i in range(len(test)-order):
      history, char = test[i:i+order], test[i+order]
      p = self.print_probs(history, char)
      if p > 0:
        perp = perp + np.log(1/p)
        N += 1
    return np.exp(perp/N)

  def save(self, filename):
    with open(filename, 'w') as file:
     file.write(json.dumps(
         {"lm": self.lm, "order": self.order, "lm_cn": self.lm_cn}))

  def load(self, filename):
    with open(filename) as f:
      data = json.load(f)
    self.order = data["order"]
    self.lm = data["lm"]
    self.lm_cn = dict(data["lm_cn"])


def calculate_prob_with_backoff(char, history, lms, lambdas):
  '''Uses interpolation to compute the probability of char given a series of 
     language models trained with different length n-grams.

   Inputs:
     char: Character to compute the probability of.
     history: A sequence of previous text.
     lms: A list of language models, outputted by calling train_char_lm.
     lambdas: A list of weights for each lambda model. These should sum to 1.
    
  Returns:
    Probability of char appearing next in the sequence.
  '''
  prob = 0
  if len(lms) != len(lambdas) or sum(lambdas) != 1:
    return None
  for i in range(len(lms)):
    prob = prob + lambdas[i]*lms[i].print_probs(history, char)
  return prob


def set_lambdas(lms, dev_filename):
  '''Returns a list of lambda values that weight the contribution of each n-gram model
  '''
  perp = []
  for lm in lms:
    perp.append(lm.perplexity(dev_filename))
  s = sum(perp)
  return [each/s for each in perp]
