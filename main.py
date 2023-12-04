# import pandas as pd
import re
# from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration
# import tokenizers
# import json
# import puz
# import os
import numpy as np
# import streamlit as st
# import scipy

# import sys
# import subprocess
# import copy
# import json

from itertools import zip_longest
from copy import deepcopy
# import regex

# from solver.Crossword import Crossword
# from solver.BPSolver import BPSolver
# from models import setup_closedbook, setup_t5_reranker, DPRForCrossword
# from solver.Utils import print_grid

# from utils import puz_to_json

import load
import models
import random
import pickle
import json

import re
import itertools
import time

DEFAULT_LEN = 5
QUERY_CAP = 50000

def clean_string(s, remove_spaces=True):
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    s = s.lower()
    if remove_spaces:
        s = re.sub(r'\s+', '', s)
    return s

def check_chars(word, fixed, fixed_len=False):
    if fixed_len and len(word) != len(fixed):
        return False
    for i, c in enumerate(word):
        if i >= len(fixed):
            return False # just for now
        if not c in fixed[i]:
            return False
    return True

def valid_cut(cut, bank):
    for i in range(len(bank)):
        if isinstance(bank[i], str) and not check_chars(bank[i], cut[i], fixed_len=True):
            return False
    return True

def merge_cuts(cuts, size, cnt, bank):
    rv = [[dict() for _ in range(size)] for _ in range(cnt)]
    for cut in cuts:
        if not valid_cut(cut, bank):
            continue
        for i, c in enumerate(cut):
            for j, chars in enumerate(c):
                for k, v in chars.items():
                    if k in rv[i][j]:
                        rv[i][j][k] = v # constant probability for now
                    else:
                        rv[i][j][k] = v
    return rv

def get_cuts(base, cnt):
    if cnt == 1:
        return [[base]]
    else:
        cuts = []
        for i in range(len(base) - cnt + 1):
            for p in get_cuts(base[i + 1:], cnt - 1):
                cuts.append([base[:i + 1]] + p)
        return cuts
    
def valid_partition(partition, bank):
    for i in range(len(bank)):
        if isinstance(bank[i], str) and partition[i] != bank[i]:
            return False
    return True

def get_partitions(source, cnt):
    if cnt == 1:
        return [[source]]
    else:
        partitions = []
        for i in range(len(source) - cnt + 1):
            for p in get_partitions(source[i + 1:], cnt - 1):
                partitions.append([source[:i + 1]] + p)
        return partitions
    
def get_container_partitions(source, cnt = 2):
    assert(cnt == 2)
    partitions = []
    for i in range(1, len(source) - 1):
        for j in range(i + 1, len(source)):
            partitions.append([source[:i] + source[j:], source[i:j]])
    return partitions

# dpr = models.setup_closedbook(0)
# with open('dpr.pkl', 'wb') as f:
#     pickle.dump(dpr, f)
dpr_cache = {}
# cw_dict = load.load_words(only_ans=True)
# with open('cw_dict.pkl', 'wb') as f:
#     pickle.dump(cw_dict, f)
with open('dpr.pkl', 'rb') as f:
    dpr = pickle.load(f)
# with open('dpr_cache.pkl', 'rb') as f:
#     dpr_cache = pickle.load(f)
with open('cw_dict.pkl', 'rb') as f:
    cw_dict = pickle.load(f)

def apply_cond(word, min_len, max_len, fixed, in_dict):
    return len(word) >= min_len \
        and len(word) <= max_len \
        and (not fixed or check_chars(word, fixed)) \
        and (not in_dict or word in cw_dict)

def synonyms(bank, min_len, max_len, fixed, in_dict):
    if isinstance(bank, list):
        bank = ' '.join(bank)
    if not bank in dpr_cache:
        print("Querying DPR for", bank)
        preds, preds_scores = models.answer_clues(dpr, [bank], QUERY_CAP, output_strings=True)
        preds, preds_scores = preds[0], list(preds_scores[0])
        preds = [clean_string(p) for p in preds]
        max_score = max(preds_scores)
        score_sum = np.sum(np.exp(np.array(preds_scores) - max_score))
        log_score_sum = np.log(score_sum) + max_score
        for i, score in enumerate(preds_scores):
            preds_scores[i] = np.exp(score - log_score_sum)
        preds_map = {}
        for i, pred in enumerate(preds):
            preds_map[pred] = preds_scores[i]
        dpr_cache[bank] = preds_map
        with open('dpr_cache.pkl', 'wb') as f:
            pickle.dump(dpr_cache, f)
    return dpr_cache[bank]

class Operator:
    def __init__(self, bank=''):
        if isinstance(bank, str):   # pure string, need to parse
            bank = clean_string(bank, remove_spaces=False)
            bank = bank.split()
            bank = [{b: 1} for b in bank]
        else:                       # list of Words and Operators, no change necessary
            pass
        self.bank = bank
        self.bank0 = deepcopy(bank)
        self.eval_factor = False
        self.factor = 0 # should never be 0
        self.op_type = "Operator"
        
    def __str__(self):
        output = self.op_type + "("
        for b in self.bank0:
            if isinstance(b, str):
                output += b + ", "
            else:
                output += str(b) + ", "
        output = output[:-2] + ")"
        return output
    
    def __repr__(self):
        return str(self)
        
    def all_banks(self, i=0):
        if i >= len(self.bank):
            yield ([], 1)
        else:
            for w, w_s in self.bank[i].items():
                if w_s == 0:
                    continue
                for b, b_s in self.all_banks(i + 1):
                    if b_s == 0:
                        continue
                    yield ([w] + b, w_s * b_s)
    
    def net_factor(self):
        if not self.eval_factor:
            for b in self.bank:
                if isinstance(b, Operator):
                    self.factor *= b.net_factor()
            self.eval_factor = True
        return self.factor
    
    def eval_full(self, min_len, max_len, fixed, in_dict):
        if fixed:
            cuts = get_cuts(fixed, len(self.bank))
            merged = merge_cuts(cuts, len(fixed), len(self.bank), self.bank)
            
        self.net_factor()
        new_bank = self.bank
        branch = []
        for i, b in enumerate(self.bank):
            if isinstance(b, Operator):
                branch.append([i, b.factor])
        branch.sort(key=lambda x: x[1])
        branch = [b[0] for b in branch]
        
        if fixed:
            for i in branch:
                new_fixed = merged[i]
                new_bank[i] = new_bank[i].eval_full(min_len, max_len, new_fixed, in_dict)
        else:
            for i in branch:
                new_bank[i] = new_bank[i].eval_full(min_len, max_len, fixed, in_dict)
                
        return {}
    
    def eval_one(self, min_len, max_len, fixed, in_dict, target=None, container=False):
        if target == None:
            if fixed:
                cuts = get_cuts(fixed, len(self.bank))
                merged = merge_cuts(cuts, len(fixed), len(self.bank), self.bank)
                
            self.net_factor()
            new_bank = self.bank
            branch = []
            for i, b in enumerate(self.bank):
                if isinstance(b, Operator):
                    branch.append([i, b.factor])
            branch.sort(key=lambda x: x[1])
            branch = [b[0] for b in branch]
            
            if fixed:
                for i in branch:
                    new_fixed = merged[i]
                    new_bank[i] = new_bank[i].eval_one(min_len, max_len, new_fixed, in_dict, None)
            else:
                for i in branch:
                    new_bank[i] = new_bank[i].eval_one(min_len, max_len, fixed, in_dict, None)
                    
            return {}
        else:
            if not container:
                partitions = get_partitions(target, len(self.bank))
            else:
                partitions = get_container_partitions(target)
            partitions = [p for p in partitions if valid_partition(p, self.bank)]
            new_bank = deepcopy(self.bank)
            for i, b in enumerate(self.bank):
                if isinstance(b, Operator):
                    new_bank[i] = {}
            for p in partitions:
                for i, b in enumerate(self.bank):
                    if isinstance(b, Operator):
                        tmp = deepcopy(b)
                        tmp = tmp.eval_one(min_len, max_len, fixed, in_dict, p[i])
                        for k, v in tmp.items():
                            if not k in new_bank[i]:
                                new_bank[i][k] = 0
                            new_bank[i][k] += v
            self.bank = new_bank
            return {}

class Alternation(Operator):
    def __init__(self, bank=''):
        super().__init__(bank)
        self.factor = 2 # O(1)
        self.op_type = "Alternation"
        
    def eval_full(self, min_len, max_len, fixed, in_dict):
        super().eval_full(min_len*2, max_len*2+1, None, False) # alternation removes letters
        self.options = {}
        for b, b_s in self.all_banks():
            merged = ''.join(b)
            evens = merged[::2]
            if apply_cond(evens, min_len, max_len, fixed, in_dict):
                if not evens in self.options:
                    self.options[evens] = 0
                self.options[evens] += b_s
            odds = merged[1::2]
            if apply_cond(odds, min_len, max_len, fixed, in_dict):
                if not odds in self.options:
                    self.options[odds] = 0
                self.options[odds] += b_s
        return self.options
    
    def eval_one(self, min_len, max_len, fixed, in_dict, target):
        if target == None:
            return self.eval_full(min_len, max_len, fixed, in_dict)
        
        super().eval_one(min_len*2, max_len*2+1, None, False) # alternation removes letters
        rv = 0
        for b, b_s in self.all_banks():
            if b_s == 0:
                continue
            merged = ''.join(b)
            evens = merged[::2]
            if evens == target:
                rv += b_s
            odds = merged[1::2]
            if odds == target:
                rv += b_s
        return {target: rv}

class Anagram(Operator):
    def __init__(self, bank=''):
        super().__init__(bank)
        total_len = 0
        for b in self.bank:
            if isinstance(b, str):
                total_len += len(b)
            else:
                total_len += DEFAULT_LEN
        self.factor = min(len(cw_dict), np.math.factorial(total_len)) # O(min(n!, len(dict)))
        self.op_type = "Anagram"
        
    def eval_full(self, min_len, max_len, fixed, in_dict):
        super().eval_full(min_len, max_len, None, in_dict)
        self.options = {}
        for b, b_s in self.all_banks():
            merged = ''.join(b)
            # if not in_dict: # waste of time
            if False:
                for perm in itertools.permutations(merged):
                    cur = ''.join(perm)
                    if apply_cond(cur, min_len, max_len, fixed, in_dict):
                        if not cur in self.options:
                            self.options[cur] = 0
                        self.options[cur] = b_s # we don't count a1a2 and a2a1 as distinct
            else:
                for word in cw_dict:
                    if len(word) == len(merged) and sorted(word) == sorted(merged) and apply_cond(word, min_len, max_len, fixed, in_dict):
                        if not word in self.options:
                            self.options[word] = 0
                        self.options[word] = b_s # we don't count a1a2 and a2a1 as distinct
        return self.options
    
    def eval_one(self, min_len, max_len, fixed, in_dict, target):
        if target == None:
            return self.eval_full(min_len, max_len, fixed, in_dict)
        
        super().eval_one(min_len, max_len, None, in_dict)
        rv = 0
        tmp = sorted(target)
        for b, b_s in self.all_banks():
            if b_s == 0:
                continue
            merged = ''.join(b)
            merged = sorted(merged)
            if merged == tmp:
                rv += b_s
        return {target: rv}
    
class Concatenation(Operator):
    def __init__(self, bank=''):
        super().__init__(bank)
        self.factor = 1 # O(1)
        self.op_type = "Concatenation"
        
    def eval_full(self, min_len, max_len, fixed, in_dict):
        super().eval_full(1, max_len, fixed, False)
        self.options = {}
        for b, b_s in self.all_banks():
            merged = ''.join(b)
            if apply_cond(merged, min_len, max_len, fixed, in_dict):
                if not merged in self.options:
                    self.options[merged] = 0
                self.options[merged] += b_s
        return self.options
    
    def eval_one(self, min_len, max_len, fixed, in_dict, target):
        if target == None:
            return self.eval_full(min_len, max_len, fixed, in_dict)
        
        super().eval_one(1, max_len, fixed, False, target)
        rv = 0
        for b, b_s in self.all_banks():
            if b_s == 0:
                continue
            merged = ''.join(b)
            if merged == target:
                rv += b_s
        return {target: rv}

class Container(Operator):
    def __init__(self, bank=''):
        super().__init__(bank)
        total_len = 0
        for b in self.bank:
            if isinstance(b, str):
                total_len += len(b)
            else:
                total_len += DEFAULT_LEN
        self.factor = total_len # O(n)
        self.op_type = "Container"
        
    def eval_full(self, min_len, max_len, fixed, in_dict):
        super().eval_full(1, 100, None, in_dict) # for now just to avoid trying to fix any characters
        self.options = {}
        for b, b_s in self.all_banks():
            assert len(b) == 2 # for now
            for _ in range(2):
                for i in range(len(b[0]) + 1):
                    cur = b[0][:i] + b[1] + b[0][i:]
                    if apply_cond(cur, min_len, max_len, fixed, in_dict):
                        if not cur in self.options:
                            self.options[cur] = 0
                        self.options[cur] += b_s
                b = [b[1], b[0]]
        return self.options
    
    def eval_one(self, min_len, max_len, fixed, in_dict, target):
        if target == None:
            return self.eval_full(min_len, max_len, fixed, in_dict)
        
        super().eval_one(1, 100, None, in_dict, target, container=True)
        rv = 0
        for b, b_s in self.all_banks():
            if b_s == 0:
                continue
            assert len(b) == 2
            if len(b[0]) + len(b[1]) != len(target):
                continue
            for i in range(len(b[0]) - 1):
                tmp = b[0][:i+1] + b[1] + b[0][i+1:]
                if tmp == target:
                    rv += b_s
        return {target: rv}

# not really sure if this is distinct to other ops..
# https://en.wikipedia.org/wiki/Cryptic_crossword#Deletions
# https://cryptics.georgeho.org/data/clues/234180
class Deletion(Operator):
    def __init__(self, bank=''):
        super().__init__(bank)
        
    def eval_full(self):
        # TODO
        raise NotImplementedError
    
class Hidden(Operator):
    def __init__(self, bank=''):
        super().__init__(bank)
        total_len = 0
        for b in self.bank:
            if isinstance(b, str):
                total_len += len(b)
            else:
                total_len += DEFAULT_LEN
        self.factor = total_len ** 2 # O(n^2)
        self.op_type = "Hidden"
        
    def eval_full(self, min_len, max_len, fixed, in_dict):
        super().eval_full(1, 100, None, in_dict) # hidden removes letters
        self.options = {}
        for b, b_s in self.all_banks():
            merged = ''.join(b)
            for i in range(len(merged)):
                for j in range(i + 1, len(merged)): # need to enforce using all words
                    cur = merged[i:j]
                    if apply_cond(cur, min_len, max_len, fixed, in_dict):
                        if not cur in self.options:
                            self.options[cur] = 0
                        self.options[cur] += b_s
        return self.options
    
    def eval_one(self, min_len, max_len, fixed, in_dict, target):
        if target == None:
            return self.eval_full(min_len, max_len, fixed, in_dict)
        
        super().eval_one(1, 100, None, in_dict)
        rv = 0
        for b, b_s in self.all_banks():
            if b_s == 0:
                continue
            merged = ''.join(b)
            for i in range(len(merged) - len(target) + 1):
                if merged[i:i+len(target)] == target:
                    rv += b_s
        return {target: rv}
    
class Initialism(Operator):
    def __init__(self, bank=''):
        super().__init__(bank)
        self.factor = 1 # O(1)
        self.op_type = "Initialism"
        
    def eval_full(self, min_len, max_len, fixed, in_dict):
        super().eval_full(1, 100, None, in_dict) # initialism removes letters
        self.options = {}
        for b, b_s in self.all_banks():
            merged = ''.join([b_[0] for b_ in b])
            if apply_cond(merged, min_len, max_len, fixed, in_dict):
                if not merged in self.options:
                    self.options[merged] = 0
                self.options[merged] += b_s
        return self.options
    
    def eval_one(self, min_len, max_len, fixed, in_dict, target):
        if target == None:
            return self.eval_full(min_len, max_len, fixed, in_dict)
        
        super().eval_one(1, 100, None, in_dict)
        rv = 0
        for b, b_s in self.all_banks():
            if b_s == 0:
                continue
            merged = ''.join([b_[0] for b_ in b])
            if merged == target:
                rv += b_s
        return {target: rv}
    
class Terminalism(Operator):
    def __init__(self, bank=''):
        super().__init__(bank)
        self.factor = 1 # O(1)
        self.op_type = "Terminalism"
        
    def eval_full(self, min_len, max_len, fixed, in_dict):
        super().eval_full(1, 100, None, in_dict) # terminalism removes letters
        self.options = {}
        for b, b_s in self.all_banks():
            merged = ''.join([b_[-1] for b_ in b])
            if apply_cond(merged, min_len, max_len, fixed, in_dict):
                if not merged in self.options:
                    self.options[merged] = 0
                self.options[merged] += b_s
        return self.options
    
    def eval_one(self, min_len, max_len, fixed, in_dict, target):
        if target == None:
            return self.eval_full(min_len, max_len, fixed, in_dict)
        
        super().eval_one(1, 100, None, in_dict)
        rv = 0
        for b, b_s in self.all_banks():
            if b_s == 0:
                continue
            merged = ''.join([b_[-1] for b_ in b])
            if merged == target:
                rv += b_s
        return {target: rv}
    
class Homophone(Operator):
    def __init__(self, bank=''):
        super().__init__(bank)
        
    def eval_full(self):
        # TODO
        raise NotImplementedError

# not really sure if this is distinct to other ops..
# https://cryptics.georgeho.org/data/clues/464087
# https://cryptics.georgeho.org/data/clues/2313
class Insertion(Operator):
    def __init__(self, bank=''):
        super().__init__(bank)
        
    def eval_full(self):
        # TODO
        raise NotImplementedError

class Reversal(Operator):
    def __init__(self, bank=''):
        super().__init__(bank)
        self.factor = 1 # O(1)
        self.op_type = "Reversal"
        
    def eval_full(self, min_len, max_len, fixed, in_dict):
        if fixed:
            super().eval_full(min_len, max_len, fixed[::-1], in_dict)
        else:
            super().eval_full(min_len, max_len, None, in_dict)
        self.options = {}
        for b, b_s in self.all_banks():
            merged = ''.join(b)[::-1]
            if apply_cond(merged, min_len, max_len, fixed, in_dict):
                if not merged in self.options:
                    self.options[merged] = 0
                self.options[merged] += b_s
        return self.options
    
    def eval_one(self, min_len, max_len, fixed, in_dict, target):
        if target == None:
            return self.eval_full(min_len, max_len, fixed, in_dict)
        
        if fixed:
            super().eval_one(min_len, max_len, fixed[::-1], in_dict, target[::-1])
        else:
            super().eval_one(min_len, max_len, None, in_dict, target[::-1])
            
        rv = 0
        for b, b_s in self.all_banks():
            if b_s == 0:
                continue
            merged = ''.join(b)[::-1]
            if merged == target:
                rv += b_s
        return {target: rv}

class Substitution(Operator):
    def __init__(self, bank=''):
        super().__init__(bank)
        self.factor = QUERY_CAP # O(QUERY_CAP)
        self.op_type = "Substitution"
        
    def eval_full(self, min_len, max_len, fixed, in_dict):
        super().eval_full(1, 100, None, in_dict) # substitution removes letters
        self.options = {}
        for b, b_s in self.all_banks():
            merged = ' '.join(b)
            syns = synonyms(merged, min_len, max_len, fixed, in_dict)
            for syn, syn_s in syns.items():
                # syn = clean_string(syn) # should already be clean
                if apply_cond(syn, min_len, max_len, fixed, in_dict):
                    if not syn in self.options:
                        self.options[syn] = 0
                    self.options[syn] += b_s * syn_s
        self.options = sorted(self.options.items(), key=lambda x: -x[1])
        self.options = self.options[:2000] # for now
        self.options = {o[0]: o[1] for o in self.options}
        return self.options
    
    def eval_one(self, min_len, max_len, fixed, in_dict, target):
        if target == None:
            return self.eval_full(min_len, max_len, fixed, in_dict)
        
        super().eval_one(1, 100, None, in_dict)
        rv = 0
        for b, b_s in self.all_banks():
            if b_s == 0:
                continue
            merged = ' '.join(b)
            syns = synonyms(merged, min_len, max_len, fixed, in_dict)
            for syn, syn_s in syns.items():
                if syn == target:
                    rv += b_s * syn_s
        return {target: rv}

class Definition:
    def __init__(self, defn):
        self.defn = defn
        
    def __str__(self):
        return "Definition(\"" + self.defn + "\")"
    
    def __repr__(self):
        return str(self)
    
    def eval_full(self, min_len, max_len, fixed, in_dict):
        return synonyms(self.defn, min_len, max_len, fixed, in_dict)
    
    def eval_one(self, min_len, max_len, fixed, in_dict, target):
        return synonyms(self.defn, min_len, max_len, fixed, in_dict)

def normalize_dict(d):
    s = sum(d.values())
    for k in d:
        d[k] /= s
    return d

def eval_full(part1, part2, ans):
    full = {}
    for c in "abcdefghijklmnopqrstuvwxyz":
        full[c] = 1
    fixed = [full] * len(ans)
    dict1 = part1.eval_full(len(ans), len(ans), fixed, True)
    dict2 = part2.eval_full(len(ans), len(ans), fixed, True)
    dict1 = normalize_dict(dict1)
    dict2 = normalize_dict(dict2)
    merged = {}
    for k1, v1 in dict1.items():
        if not k1 in dict2:
            continue
        merged[k1] = v1 * dict2[k1]
    merged = normalize_dict(merged)
    return merged.get(ans, 0)

def eval_one(part1, part2, ans, log=True):
    if log:
        print("eval_one(" + str(part1) + ", " + str(part2) + ", \"" + ans + "\")")
    start_time = time.time()
    dict1 = part1.eval_one(len(ans), len(ans), None, True, ans)
    dict2 = part2.eval_one(len(ans), len(ans), None, True, ans)
    end_time = time.time()
    term1 = dict1.get(ans, 0)
    term2 = dict2.get(ans, 0)
    if log:
        print("Time:", end_time - start_time)
        print(term1, "*", term2, "=", term1 * term2)    
    return term1 * term2

# # Speak about idiot making sense (6)
# # Container([Substitution("Speak") [about], Substitution("idiot")]) [making], Definition("sense"), "sanity"
# eval_one(Container([Substitution("Speak"), Substitution("idiot")]), Definition("sense"), "sanity")
# # Concatenation([Substitution("A long arduous journey, especially one made on foot."), Substitution("chess piece")]), Definition("Walking"), "trekking"
# eval_one(Concatenation([Substitution("A long arduous journey, especially one made on foot."), Substitution("chess piece")]), Definition("Walking"), "trekking")
# # Anagram("to a smart set"), Definition("Provider of social introductions"), "toastmaster"
# eval_one(Anagram("to a smart set"), Definition("Provider of social introductions"), "toastmaster")
# # Anagram("to a smart set"), Definition("Provider of social introductions"), "greeter"
# eval_one(Anagram("to a smart set"), Definition("Provider of social introductions"), "greeter")
# # Odd stuff of Mr. Waugh is set for someone wanting women to vote (10)
# # [Odd] Alternation("stuff of Mr. Waugh is set"), [for] Definition("someone wanting women to vote"), "suffragist"
# eval_one(Alternation("stuff of Mr. Waugh is set"), Definition("someone wanting women to vote"), "suffragist")
# # Outlaw leader managing money (7)
# # Concatenation([Substitution("Outlaw"), Substitution("leader")]), Definition("managing money"), "banking"
# eval_one(Concatenation([Substitution("Outlaw"), Substitution("leader")]), Definition("managing money"), "banking")
# # Country left judgeable after odd losses (8)
# # Definition("Country"), Concatenation([Substitution("left"), Alternation("judgeable")]) [after odd losses], "portugal"
# eval_one(Definition("Country"), Concatenation([Substitution("left"), Alternation("judgeable")]), "portugal")
# # Shade’s a bit circumspect, really (7)
# # Definition("Shade's"), [a bit] Hidden("circumspect, really"), "spectre"
# eval_one(Definition("Shade's"), Hidden("circumspect, really"), "spectre")
# # A bit of god-awful back trouble (3)
# # [A bit of] Reversal([Hidden("god-awful")]) [back], Definition("trouble"), "ado"
# eval_one(Reversal([Hidden("god-awful")]), Definition("trouble"), "ado")
# # Quangos siphoned a certain amount off, creating scandal (6)
# # Hidden("Quangos siphoned") [a certain amount off], Definition("creating scandal"), "gossip"
# eval_one(Hidden("Quangos siphoned"), Definition("creating scandal"), "gossip")
# # Bird is cowardly, about to fly away (5)
# # Definition("Bird"), [is] Hidden([Substitution("cowardly,")]) [about to fly away], "raven"
# eval_one(Definition("Bird"), Hidden([Substitution("cowardly,")]), "raven")
# # As is a less stimulating cup defeat, faced in a bad way (13)
# # Definition("As is a less stimulating cup"), Anagram("defeat, faced in") [a bad way], "decaffeinated"
# eval_one(Definition("As is a less stimulating cup"), Anagram("defeat, faced in"), "decaffeinated")
# # At first, actor needing new identity emulates orphan in musical theatre (5)
# # [At first], Initialism("actor needing new identity emulates"), Definition("orphan in musical theatre"), "annie"
# eval_one(Initialism("actor needing new identity emulates"), Definition("orphan in musical theatre"), "annie")
# # Bird with tips of rich aqua, yellow, black (4)
# # Definition("Bird"), [with tips of] Terminalism("rich aqua, yellow, black"), "hawk"
# eval_one(Definition("Bird"), Terminalism("rich aqua, yellow, black"), "hawk")

ops = {
    "Alternation": Alternation,
    "Anagram": Anagram,
    "Concatenation": Concatenation,
    "Container": Container,
    "Hidden": Hidden,
    "Initialism": Initialism,
    "Reversal": Reversal,
    "Substitution": Substitution,
    "Terminalism": Terminalism,
}

adj = {
    Alternation: [Anagram, Concatenation, Container, Hidden, Initialism, Reversal, Substitution, Terminalism],
    Anagram: [Alternation, Concatenation, Container, Hidden, Initialism, Reversal, Substitution, Terminalism],
    Concatenation: [Alternation, Anagram, Container, Hidden, Initialism, Reversal, Substitution, Terminalism],
    Container: [Alternation, Anagram, Concatenation, Hidden, Initialism, Reversal, Substitution, Terminalism],
    Hidden: [Alternation, Anagram, Concatenation, Container, Initialism, Reversal, Substitution, Terminalism],
    Initialism: [Alternation, Anagram, Concatenation, Container, Hidden, Reversal, Substitution, Terminalism],
    Reversal: [Alternation, Anagram, Concatenation, Container, Hidden, Initialism, Substitution, Terminalism],
    Substitution: [Alternation, Anagram, Concatenation, Container, Hidden, Initialism, Reversal, Terminalism],
    Terminalism: [Alternation, Anagram, Concatenation, Container, Hidden, Initialism, Reversal, Substitution],
}

short_clues = load.load_data()
short_clues = [c for c in short_clues if len(c[0].split()) <= 4 and len(c[0].split()) >= 2]
print(len(short_clues))
random.shuffle(short_clues)

def gen_ops(words, cur_ops=list(ops.values()), i=0, depth=1, rec_p=0.3, max_depth=3, exit_p=0.3):
    cur_op = np.random.choice(cur_ops)
    new_ops = [o for o in adj[cur_op] if o in cur_ops]
    bank = []
    while i < len(words):
        if np.random.rand() < rec_p and depth < max_depth and len(new_ops) > 0:
            new_op, i = gen_ops(words, new_ops, i, depth + 1, rec_p, max_depth, exit_p)
            bank.append(new_op)
        else:
            bank.append({words[i]: 1})
            i += 1
        if np.random.rand() < exit_p and depth > 1:
            break
        if cur_op == Container and len(bank) == 2:
            break
    while cur_op == Container and len(bank) == 1:
        cur_op = np.random.choice(cur_ops)
    if depth == 1:
        return cur_op(bank)
    else:
        return cur_op(bank), i

def gen_all(clue, del_p=0.1):
    words = clue.split()
    defn_len = np.random.randint(-len(words) + 2, len(words))
    if defn_len <= 0:
        defn_len -= 1
        defn = ' '.join(words[defn_len:])
        nondefn = words[:defn_len]
    else:
        defn = ' '.join(words[:defn_len])
        nondefn = words[defn_len:]
    tmp = []
    while len(tmp) == 0:
        tmp = [w for w in nondefn if np.random.rand() > del_p]
    nondefn = tmp
    return Definition(defn), gen_ops(nondefn)

def try_clue(clue, ans):
    defn, ops = gen_all(clue)
    return eval_one(defn, ops, ans, log=False), defn, ops
    
# runs = []
with open('runs.json', 'r') as f:
    runs = json.load(f)

for clue, nondef, defn, ans, sz in short_clues:
    print("Solving clue: " + clue + " (" + ans + ")")
    dpr_cache = {}
    best = (0, None, None)
    tried = []
    for _ in range(100):
        cur = try_clue(clue, ans)
        cur = (cur[0], str(cur[1]), str(cur[2]))
        tried.append(cur)
        if cur[0] > best[0]:
            best = cur
    print(best)
    print()
    runs.append({
        "clue": clue,
        "nondef": nondef,
        "defn": defn,
        "ans": ans,
        "sz": sz,
        "best": best,
        "tried": tried
    })
    with open('runs.json', 'w') as f:
        json.dump(runs, f)