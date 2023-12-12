import numpy as np
import random
import copy

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import models
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM
import itertools

dpr = models.setup_closedbook(0)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

class CrosswordEnv():
    # self.full_data:
    # [(nondef: str, defn: str, ans: str, sz: int), ...]
    
    def __init__(self, data):
        self.tokenizer = tokenizer # TODO: delete
        self.full_data = data
        self.BATCH_SIZE = 20
        self.PROMPT = PROMPT = "Find the next state to solve the cryptic crossword. Do not stop unless state has the right LENGTH. DEFINITION {definition} LENGTH {length} {stop} CLUE {clue} | STEPS {steps} STATE {state}\n\nNEXTSTATE"
    
    def get_reward(self):
        # definition similarity
        new_score = np.empty(self.done.shape)
        if len(self.scores) > 0:
            new_score = self.scores[-1]
        new_score[~self.done] = dpr.get_scores(list(itertools.compress(self.acs_str, list(~self.done))), list(itertools.compress(self.targets, list(~self.done))))
        # length similarity
        for i, s in enumerate(self.acs_str):
            new_score[i] -= (len(s) - len(self.targets[i])) ** 2
            # equality
            new_score += (1 if s == self.targets[i] else 0) * 1e3
            if not self.done[i] and s == "STOP" and len(self.targets[i]) != len(self.prev[i]):
                #penalize for stopping without matching the length
                new_score[i] -= 1e3
        # character similarity
        for i, s in enumerate(self.acs_str):
            if self.done[i]:
                continue
            alpha_c = np.zeros(26)
            for c in s:
                if ord(c) -ord('a') >= 26 or ord(c) -ord('a') < 0:
                    continue
                if c == " ":
                    continue
                alpha_c[ord(c) - ord('a')] += 1
            for c in self.targets[i]:
                if ord(c) -ord('a') >= 26 or ord(c) -ord('a') < 0:
                    continue
                if c == " ":
                    continue
                alpha_c[ord(c) - ord('a')] -= 1
            new_score[i] -= np.sum(alpha_c ** 2)
        
        self.scores.append(new_score)
        self.prev = self.acs_str
        print(self.acs_str, self.targets)
        print(np.mean(new_score))
        if len(self.scores) > 1:
            return self.scores[-1] - self.scores[-2]
        else:
            return new_score
    
    def reset(self):
        self.cur_data = random.sample(self.full_data, self.BATCH_SIZE)
        self.cur_step = 0
        self.obs_str = []
        self.acs_str = []
        self.scores = []
        self.lens = []
        self.targets = []
        self.prev = []
        self.done = np.array([False] * len(self.cur_data))
        for nondef, defn, ans, sz in self.cur_data:
            self.obs_str.append(self.PROMPT.format(definition=defn, length=sz, stop=False, clue=nondef, steps=self.cur_step, state=nondef))
            self.acs_str.append(nondef)
            self.lens.append(len(self.obs_str[-1]))
            self.targets.append(ans)
            self.prev.append(nondef)
        self.get_reward()
        self.tokenized = tokenizer(self.obs_str, padding="max_length", truncation=True, return_tensors="pt", max_length=75).input_ids.cuda()
        return self.tokenized.cpu().numpy()
    
    def step(self, action):
        #tokenizer.batch_decode(tensor, skip_special_tokens=True)
        self.acs_str = tokenizer.batch_decode(action, skip_special_tokens=True)
        self.acs_str = [(s.split("\n\nNEXTSTATE ")+[""])[1] for s in self.acs_str]
        # self.acs_str = [s[self.lens[i]:] for i, s in enumerate(self.acs_str)]
        reward = self.get_reward()
        self.cur_step += 1
        if self.cur_step >= 10:
            self.done = [True] * len(self.cur_data)
        else:
            self.done = [s == "STOP" for s in self.acs_str]
        self.done = np.array(self.done)
        self.info = {}
        for i, (nondef, defn, ans, sz) in enumerate(self.cur_data):
            self.obs_str[i] = self.PROMPT.format(definition=defn, length=sz, stop=self.done[i], clue=nondef, steps=self.cur_step, state=self.acs_str[i])
        self.tokenized = tokenizer(self.obs_str, padding="max_length", truncation=True, return_tensors="pt", max_length=75).input_ids.cuda()
        return self.tokenized.cpu().numpy(), reward, self.done, self.info
        
    def close(self):
        pass
