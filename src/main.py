import math
import torch
from torch import nn
from transformers import BertTokenizer
from transformers import AdamW, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import os
import pandas as pd
import wget
from BertForTagging import *
import random
import numpy as np

if(__name__ == '__main__'):
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

    print(f'length(data): {df.shape[0]}')
    print(df.sample(3))

    statements = df.problem_statement[0:100]
    tags = df.problem_tags[0:100]

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

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    batch_size = 8

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
        problem_type = 'multi_label_classification'
        #classifier = classifier,
        #loss_module = BPMLLLoss
    )

    #model.cuda()
    if(with_gpu):
        model.cuda()
    else:
        model.cpu()

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

    optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                )
    
    epochs = 2

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    
    import time
    import datetime
    def flat_accuracy(logits, labels):
        preds = nn.Sigmoid()(logits)
        dist = torch.abs(preds - labels)
        return 1 - torch.mean(dist)

    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

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

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)

        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
        

        print("")
        print("Running Validation...")

        t0 = time.time()
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader:
            b_input_ids = batch[0].squeeze(1).to(device)
            b_input_mask = batch[1].squeeze(1).to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():        
                (loss, logits) = model(input_ids=b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().squeeze(0)
            labels = b_labels.to('cpu')

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, labels)
            

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))