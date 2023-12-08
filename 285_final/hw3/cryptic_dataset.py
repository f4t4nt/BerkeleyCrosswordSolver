import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM

assert torch.cuda.is_available()
device = torch.device("cuda")
print("Using device:", device)

from transformers import AutoTokenizer

class CrypticDataset(Dataset):
    PROMPT = "Find the next state to solve the cryptic crossword. Do not stop unless state has the right LENGTH. DEFINITION {definition} LENGTH {length} {stop} CLUE {clue} | STEPS {steps} STATE {state}\n\nNEXTSTATE {next_state}"

    def __init__(self, data):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.data = data
        self.tokenizer = self.tokenizer
        self.tokenizer.padding_side = 'right'

        self.input_ids = []
        self.attn_masks = []
        self.labels = []

        training_texts = []
        for example in self.data:
            steps = example["steps"]
            for i in range(len(steps)):
                training_text = CrypticDataset.PROMPT.format(definition=example['definition'], length=example["length"], stop=str(len(example["steps"][i])==example["length"]),
                                                             clue=example['clue'], steps=i, state=steps[i], next_state=("STOP" if i == len(example["steps"])-1 else steps[i+1])) + "<|endoftext|>" # include the end token so model knows when to stop!
                training_texts.append(training_text)
        encodings_dict = self.tokenizer(training_texts, padding=True, truncation=True)
        for i,  training_text in enumerate(training_texts):
            self.input_ids.append(torch.tensor(encodings_dict['input_ids'][i]))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask'][i]))
            prompt_and_input_length = len(self.tokenizer.encode(training_text.split("NEXTSTATE")[0]+"NEXTSTATE"))
            if i == 4:
                print("{}".format(self.tokenizer.decode(encodings_dict['input_ids'][i], skip_special_tokens=True)))
                print(encodings_dict['input_ids'][i])
                print("{}".format(self.tokenizer.decode(self.tokenizer.encode(training_text.split("NEXTSTATE")[0]+"NEXTSTATE"), skip_special_tokens=True)))
                print(self.tokenizer.encode(training_text.split("NEXTSTATE")[0]+"NEXTSTATE"))
                print(torch.tensor([-100] * prompt_and_input_length + encodings_dict['input_ids'][i][prompt_and_input_length:]))
            self.labels.append(torch.tensor([-100] * prompt_and_input_length + encodings_dict['input_ids'][i][prompt_and_input_length:]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attn_masks[idx], 'labels': self.labels[idx]}