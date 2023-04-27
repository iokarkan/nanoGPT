import json
import glob
import os
import numpy as np
import tiktoken

# # Find all the *_merged.tex files in the directory
# tex_files = glob.glob(os.path.join(papers_directory, '*_merged_utf-8.tex'))

# # Open a new file called input.tex in the same directory to append the contents of the other files
# with open(os.path.join(papers_directory, "..", 'input.tex'), 'w', encoding='utf-8', errors='ignore') as output_file:

def convert_json_to_text(out='input.text'):
    json_files = glob.glob(os.path.join("data/jokes/datasets", 'reddit_jokes.json'))
    data = []
    for f in json_files:
        print(f)
        with open(f, 'r') as infile:
            data.append(json.load(infile))

    with open(os.path.join("data/jokes/datasets", "..", out),'w') as outfile:
        for f in data:
            for item in f:
                # print(item)
                try:
                    if len(item['body'])<200:
                        outfile.write(f"- {item['title']}\n")
                        outfile.write(f"- {item['body']}\n")
                        outfile.write("\n***\n\n")
                except:
                    continue


convert_json_to_text('input.text')

input_file_path = os.path.join(os.path.dirname(__file__), 'input.text')

with open(input_file_path, 'r', errors='ignore') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))