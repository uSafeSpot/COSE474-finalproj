import math
import torch
import transformers
from transformers import BertTokenizer
from transformers import BertForMultipleChoice, AdamW, BertConfig
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import os
import pandas as pd
import wget


if(__name__ == '__main__'):
    '''if torch.cuda.is_available():    
        device = torch.device("cuda")
        print("cuda")
    else:
        device = torch.device("cpu")
        print('cpu')'''

    url = 'https://raw.githubusercontent.com/uSafeSpot/COSE474-finalproj/main/data/data.csv'

    if not os.path.exists('./data/data.csv'):
        wget.download(url, './data/data.csv')
    
    df = pd.read_csv('./data/data.csv', delimiter=',', header=0)

    print(f'length(data): {df.shape[0]}')
    print(df.sample(3))

    statements = df.problem_statement[0:10]
    tags = df.problem_tags[0:10]

    tag_set = set()

    i = 0
    for tag in tags:
        tag = tag.split(',')
        for t in tag:
            if(not t.startswith('*')): #to remove rating(difficulty) tag
                tag_set.add(t)
        i += 1

    tag_list = list(tag_set)
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
    print(f"tag space dimension: {tag_dim}")

    tk = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    tokenized = []
    att_masks = []

    for stm in statements:
        encoded = tk.encode_plus(
            stm,
            add_special_tokens = True,
            max_length = 512,
            pad_to_max_length=True,
            return_attention_mask = True,
            return_tensors = 'pt'
        )

        tokenized.append(encoded['input_ids'])
        att_masks.append(encoded['attention_mask'])

    tokenized = torch.stack(tokenized)
    att_masks = torch.stack(att_masks)

    tag_vectors = torch.stack(tag_vectors)
    
    dataset = TensorDataset(tokenized, att_masks, tag_vectors)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    batch_size = 32

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order. 
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
    
    
    
    model = BertForMultipleChoice.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = tag_dim, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    # Tell pytorch to run this model on the GPU.
    #model.cuda()
    model.cpu()

    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))