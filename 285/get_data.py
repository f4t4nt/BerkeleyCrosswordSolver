import os
import sys

print(os.getcwd())
sys.path.append(os.getcwd())

import re
import json
import pickle

if True:
    with open("./285/pkl/georgeho.pkl", "rb") as f:
        data = pickle.load(f)
        
    print("Loaded pickle")
else:
    data = []
    
    def parse_file(json_data):
        for row in json_data["rows"]:
            clue0 = row[1]
            ans = row[2]
            defn = row[3]
            if not clue0 or not ans or not defn:
                continue
            clue = clue0.lower()
            ans = ans.lower()
            defn = defn.lower()
            if "(" in clue:
                clue = clue[:clue.rindex("(")]
            # make clue alphanumeric
            clue = re.sub(r'[^a-zA-Z0-9\s]', '', clue)
            clue = clue.replace(defn, "")
            clue = clue.strip()
            data.append((clue, ans, clue0))

    def parse_george_ho():
        for file in os.listdir("data/georgeho/"):
            file_path = os.path.join("data/georgeho/", file)
            with open(file_path, "r") as f:
                json_data = json.load(f)
                parse_file(json_data)
    
    parse_george_ho()
    
    with open("./285/pkl/georgeho.pkl", "wb") as f:
        pickle.dump(data, f)