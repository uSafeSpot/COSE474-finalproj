import math
import torch
from torch import nn
import transformers
from transformers import BertTokenizer
from transformers import AdamW, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import os
import pandas as pd
import random
import numpy as np

import time
import datetime

batch_size = 8
epochs = 6
dropout_p = 0.5
learning_rate = 5e-6

with_gpu = True
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print("cuda")
else:
    device = torch.device("cpu")
    with_gpu = False
    print('cpu')

url = 'https://raw.githubusercontent.com/uSafeSpot/COSE474-finalproj/main/data/data.csv'

if not os.path.exists('./data/data.csv'):
    wget.download(url, './data/data.csv')

df = pd.read_csv('./data/data.csv', delimiter=',', header=0)

path = './data/data.csv'
save_path = './model'

#PREPROCESSING


DATA_CLIP = 0

df = pd.read_csv(path, delimiter=',', header=0)

print(f'length(data): {df.shape[0]}')
print(df.sample(3))

DATA_CLIP = df.shape[0] if not DATA_CLIP else DATA_CLIP

statements = df.problem_statement[0:DATA_CLIP]
tags = df.problem_tags[0:DATA_CLIP]

tag_cnt = dict()

i = 0
for tag in tags:
    tag = tag.split(',')
    for t in tag:
        if(not t.startswith('*')):
            if(t in tag_cnt):
                tag_cnt[t] += 1
            else:
                tag_cnt[t] = 1
    i += 1

tag_list = list()

print(tag_cnt)

for k, v in tag_cnt.items():
    if (v > 1000):
        tag_list.append(k)

tag_list = ['implementation', 'math', 'greedy', 'dp', 
            'constructivealgorithms', 'datastructures', 'bruteforce']

tag_dict = dict([(t, i) for i, t in enumerate(tag_list)])
tag_vectors = []
tag_dim = len(tag_list)
for tag in tags:
    tag = tag.split(',')
    tagvec = torch.zeros(tag_dim)
    for t in tag:
        if(t in tag_dict):
            tagvec[tag_dict[t]] = 1
    tag_vectors.append(tagvec)

tag_vectors = torch.stack(tag_vectors)

print(f"tag space dimension: {tag_dim}")
print(tag_dict)

tag_dict = {'implementation': 0, 'math': 1, 'greedy': 2, 'dp': 3, 
            'constructivealgorithms': 4, 'datastructures': 5, 'bruteforce': 6}


tk = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

tokenized = []
att_masks = []

for stm in statements:
    encoded = tk.encode_plus(
        stm,
        add_special_tokens = True,
        max_length = 512,
        pad_to_max_length=True,
        truncation=True,
        return_attention_mask = True,
        return_tensors = 'pt'
    )

    tokenized.append(encoded['input_ids'])
    att_masks.append(encoded['attention_mask'])

tokenized = torch.stack(tokenized)
att_masks = torch.stack(att_masks)

#TRAINING

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print("cuda")
else:
    device = torch.device("cpu")
    with_gpu = False
    print('cpu')

dataset = TensorDataset(tokenized, att_masks, tag_vectors)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size
            )

validation_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset),
            batch_size = batch_size
            )

classifier = nn.Linear(768, tag_dim)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = tag_dim,
    output_attentions = False,
    output_hidden_states = False,
    return_dict=False,
    problem_type = 'multi_label_classification',
    classifier_dropout = dropout_p
)

if(with_gpu):
    model.cuda()
else:
    model.cpu()

optimizer = AdamW(model.parameters(),
              lr = learning_rate,
              eps = 1e-8
            )

total_steps = len(train_dataloader) * epochs

scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

seed_val = 1

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

training_stats = []

total_t0 = time.time()

for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('[Training]')

    t0 = time.time()

    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        b_input_ids = batch[0].squeeze(1).to(device)
        b_input_mask = batch[1].squeeze(1).to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        (loss, logits) = model(input_ids=b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)

        total_train_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)

    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
    

    print("")
    print("[Running Validation]")

    t0 = time.time()
    model.eval()

    total_eval_loss = 0
    nb_eval_steps = 0

    f1_thr = torch.zeros([100])

    s_TP_thr = torch.zeros([100, tag_dim])
    s_FP_thr = torch.zeros([100, tag_dim])
    s_FN_thr = torch.zeros([100, tag_dim])

    for batch in validation_dataloader:
        b_input_ids = batch[0].squeeze(1).to(device)
        b_input_mask = batch[1].squeeze(1).to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():        
            (loss, logits) = model(input_ids=b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask,
                                labels=b_labels)
            
        total_eval_loss += loss.item()

        logits = logits.detach().cpu().squeeze(0)
        labels = b_labels.cpu().type('torch.LongTensor')

        preds = nn.Sigmoid()(logits)
        
        for _thr in range(100):
            thr = _thr / 100
            preds_b = preds > thr
            s_TP_thr[_thr] += torch.sum(preds_b & labels, dim=0)
            s_FP_thr[_thr] += torch.sum(preds_b & (1-labels), dim=0)
            s_FN_thr[_thr] += torch.sum(labels & ~preds_b, dim=0)
        
    prec = s_TP_thr / (s_TP_thr + s_FP_thr)
    recl = s_TP_thr / (s_TP_thr + s_FN_thr)

    f1_thr = 2 * prec * recl / (prec + recl)

    val_f1_thr = torch.nan_to_num(torch.mean(f1_thr, dim=1), nan=-float('inf'))
    val_f1 = torch.max(val_f1_thr)
    max_thres = torch.argmax(val_f1_thr) / 100

    print("  F1: {0:.2f}".format(val_f1*100))
    print("  threshold: {0:.2f}".format(max_thres))

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. F1.': val_f1,
            'Valid. argmax thres.': max_thres,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )
print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

model.save_pretrained(save_path)

print("Training saved!")

#PLOT

import pandas as pd

df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')

print(df_stats)

import matplotlib.pyplot as plt

import seaborn as sns

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

# Plot the learning curve.
plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

# Label the plot.
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8])

plt.show()

model = BertForSequenceClassification.from_pretrained(save_path)
print(model)

model.to('cuda')

statement = input()

tk = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
encoded = tk.encode_plus(
    statement,
    add_special_tokens = True,
    max_length = 512,
    pad_to_max_length=True,
    truncation=True,
    return_attention_mask = True,
    return_tensors = 'pt'
)
print(tk.decode(encoded['input_ids'][0]))

with torch.no_grad():
    (loss, logits) = model(input_ids=encoded['input_ids'].to('cuda'), 
                    token_type_ids=None, 
                    attention_mask=encoded['attention_mask'].to('cuda'), 
                    labels=torch.zeros([1, tag_dim]).to('cuda'))

pred = torch.nn.Sigmoid()(logits)[0] > max_thres
pred = pred.cpu()
for i in range(tag_dim):
    if(pred[i]): print(tag_list[i])