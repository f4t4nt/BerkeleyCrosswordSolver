import numpy as np
import random
import copy

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import models
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM

dpr = models.setup_closedbook(0)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

class CrosswordEnv():
    # self.full_data:
    # [(nondef: str, defn: str, ans: str, sz: int), ...]
    
    def __init__(self, data):
        self.tokenizer = tokenizer # TODO: delete
        self.full_data = data
        self.PROMPT = "Find the next state to solve the cryptic crossword. Do not stop unless state has the correct LENGTH. DEFINITION {definition} LENGTH {length} {stop} CLUE {clue} | STEPS {steps} STATE {state}\n\nNEXTSTATE {next_state}"
    
    def get_reward(self):
        score = dpr.get_scores(self.acs_str, self.targets)
        for i, s in enumerate(self.acs_str):
            score[i] -= (len(s) - len(self.targets[i])) ** 2
        for i, s in enumerate(self.acs_str):
            alpha_c = np.zeros(26)
            for c in s:
                if c == " ":
                    continue
                alpha_c[ord(c) - ord('a')] += 1
            for c in self.targets[i]:
                if c == " ":
                    continue
                alpha_c[ord(c) - ord('a')] -= 1
            score[i] -= np.sum(alpha_c ** 2)
        return score
    
    def reset(self):
        self.cur_data = random.sample(self.full_data, 1000)
        self.cur_step = 0
        self.obs_str = []
        self.lens = []
        self.targets = []
        for nondef, defn, ans, sz in self.cur_data:
            self.obs_str.append(self.PROMPT.format(definition=defn, length=sz, stop=False, clue=nondef, steps=self.cur_step, state=nondef, next_state=""))
            self.lens.append(len(self.obs_str[-1]))
            self.targets.append(ans)
        self.tokenized = tokenizer(self.obs_str, padding=True, truncation=True, return_tensors="pt").input_ids.cuda()
        return self.tokenized
    
    def step(self, action):
        #tokenizer.batch_decode(tensor, skip_special_tokens=True)
        self.acs_str = tokenizer.batch_decode(action, skip_special_tokens=True)
        # acs_str = [s.split("NEXTSTATE ")[1] for s in acs_str]
        self.acs_str = [s[self.lens[i]:] for i, s in enumerate(self.acs_str)]
        self.cur_step += 1
        done = self.cur_step >= 10
        info = {}
        for i, (nondef, defn, ans, sz) in enumerate(self.cur_data):
            self.obs_str[i] = self.PROMPT.format(definition=defn, length=sz, stop=str(len(self.acs_str[i])==sz), clue=nondef, steps=self.cur_step, state=self.acs_str[i], next_state="")
        self.tokenized = tokenizer(self.obs_str, padding=True, truncation=True, return_tensors="pt").input_ids.cuda()
        return self.tokenized, self.get_reward(), done, info
        
    def close(self):
        pass
