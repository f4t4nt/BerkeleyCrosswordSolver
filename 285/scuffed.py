
import pandas as pd
import re
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration
import tokenizers
import json
import puz
import os
import numpy as np
import streamlit as st
import scipy

import sys
import subprocess
import copy
import json

from itertools import zip_longest
from copy import deepcopy
import regex

import os
import sys

print(os.getcwd())
sys.path.append(os.getcwd())
assert os.path.exists("./solver/")

from solver.Crossword import Crossword
from solver.BPSolver import BPSolver
from models import setup_closedbook, setup_t5_reranker, DPRForCrossword
from solver.Utils import print_grid

from utils import puz_to_json

import load
import models
import random

import re
import itertools
import time
import pickle

dpr = models.setup_closedbook(0)
cw_dict = load.load_words(only_ans=True)
with open("./285/pkl/georgeho.pkl", "rb") as f:
    data = pickle.load(f)

def anagram(words, i):
    anagrams = []
    for key in cw_dict:
        if sorted(key) == sorted(words[i]) and key != words[i]:
            anagrams.append(key)
    if len(anagrams) > 0:
        words[i] = random.choice(anagrams)
    return words

def alternation(words, i):
    alternations = []
    for j in range(2):
        alternations.append("".join([words[i][k] for k in range(j, len(words[i]), 2)]))
    if len(alternations) > 0:
        words[i] = random.choice(alternations)
    return words

def container(words, i): # requires 2
    if i == len(words) - 1:
        return words
    containers = []
    for j in range(len(words[i])):
        new_word = words[i][:j] + words[i+1] + words[i][j:]
        if new_word in cw_dict:
            containers.append(new_word)
    if len(containers) > 0:
        words[i] = random.choice(containers)
        words = erase(words, i+1)
    return words
    
def erase(words, i):
    return words[:i] + words[i+1:]

def concat(words, i): # requires 2
    if i == len(words) - 1:
        return words
    concats = [words[i] + words[i+1]]
    if len(concats) > 0:
        words[i] = random.choice(concats)
        words = erase(words, i+1)
    return words

def hidden(words, i):
    hiddens = []
    for j in range(len(words[i])):
        for k in range(j+1, len(words[i])):
            new_word = words[i][:j] + words[i][j+1:k] + words[i][k+1:]
            if new_word in cw_dict:
                hiddens.append(new_word)
    if len(hiddens) > 0:
        words[i] = random.choice(hiddens)
    return words

def merge(words, i): # requires 2
    if i == len(words) - 1:
        return words
    merges = [words[i] + " " + words[i+1]]
    if len(merges) > 0:
        words[i] = random.choice(merges)
        words = erase(words, i+1)
    return words

def reversal(words, i):
    reversals = []
    reversals.append(words[i][::-1])
    if len(reversals) > 0:
        words[i] = random.choice(reversals)
    return words

def synonym(words, i):
    # TODO
    return words

ops = [anagram, alternation, container, concat, hidden, merge, reversal, synonym]

def get_score_seq(words, ans):
    try:
        total_its = random.randint(10, 20)
        phrases = [' '.join(words)]
        for i in range(total_its):
            op = random.choice(ops)
            words = op(words, random.randint(0, len(words)-1))
            phrase = ' '.join(words)
            if phrase not in phrases:
                phrases.append(phrase)
        scores = dpr.get_scores(phrases, ans)
        if type(scores) == type(None):
            return None
        scores = [float(s) for s in scores]
        scores = sorted(scores)
        for i in range(5):
            idx1 = random.randint(0, len(scores)-1)
            idx2 = random.randint(0, len(scores)-1)
            scores[idx1], scores[idx2] = scores[idx2], scores[idx1]
        return scores
    except:
        return None
""
random.shuffle(data)
with open("./285/pkl/all_runs.pkl", "rb") as f:
    all_runs = pickle.load(f)

for i, (clue, ans, clue0) in enumerate(data):
    if i % 10 == 0:
        with open("./285/pkl/all_runs.pkl", "wb") as f:
            pickle.dump(all_runs, f)
    words = clue.split()
    scores = get_score_seq(words, ans)
    if scores != None:
        all_runs.append(scores)