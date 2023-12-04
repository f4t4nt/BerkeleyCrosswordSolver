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

dpr = models.setup_closedbook(0)
print(type(dpr.retriever.index.index))
d = dpr.retriever.index.index.d
print(d)

vec = np.empty((1, d), dtype='float32')

# Call the reconstruct method
dpr.retriever.index.index.reconstruct(0, vec[0])

test_qs = [
    "What is the capital of France?",
    "Who is the president of the United States?",
    "Who wrote the Harry Potter books?",
    "What is the most popular sport in the US?",
    "What is the most popular sport in the world?",
    "This is just a test",
    "What happens if I put another",
    "That wouldnt be fun",
    "I think so",
    "What else can I say",
    "This is just a test",
    "That's two of the same",
    "Something new here",
    "Filler",
    "Almost there",
    "Okay, this should be good"
]

start_time = time.time()
# print(vec[0])
tmp = dpr.get_scores(test_qs, "justtesting")
end_time = time.time()
print(end_time - start_time)
print(tmp)

start_time = time.time()
preds, preds_scores = models.answer_clues(dpr, test_qs, 500000, output_strings=True)
end_time = time.time()
print(end_time - start_time)

for i, pred in enumerate(preds):
    for j, p in enumerate(pred):
        if p == "JUSTTESTING":
            print(preds_scores[i][j])

test_phrases = ["meal hasnt started", "hasnt started", "hasnt", "meal started", "dinner hasnt started", "dinner started", "dinner start", "started", "dinner", "inner"]
print(dpr.get_scores(test_phrases, "inner"))