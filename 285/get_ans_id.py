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

dpr = models.setup_closedbook(0)

import pickle

if True:
    with open("./285/pkl/all_passages.pkl", "rb") as f:
        all_passages = pickle.load(f)
        
    with open("./285/pkl/all_passages_inv.pkl", "rb") as f:
        all_passages_inv = pickle.load(f)
        
    print("Loaded pickles")
else: 
    all_passages = dpr.all_passages

    with open("./285/pkl/all_passages.pkl", "wb") as f:
        pickle.dump(all_passages, f)
        
    all_passages_inv = {}
    for key, value in all_passages.items():
        all_passages_inv[value.lower()] = int(key)

    with open("./285/pkl/all_passages_inv.pkl", "wb") as f:
        pickle.dump(all_passages_inv, f)