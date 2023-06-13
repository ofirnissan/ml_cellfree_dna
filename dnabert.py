# Playing With DNABERT:
    # At this script we tried to print out embeddings
    # Also, we tried to use the model as is, in order to classify (male / female)

import torch
from transformers import BertModel, BertTokenizer, BertForSequenceClassification

# Choose one of: 'zhihan1996/DNA_bert_$k' as 3 <= k <= 6
model_name = 'zhihan1996/DNA_bert_6'

# PROCESS DATA:
# Set BertTokenizer pretrained dnabert model:
tokenizer = BertTokenizer.from_pretrained(model_name)

# DNA long sample - choose "female.tsv"/"male.tsv"
f = open("female.tsv", "r")
dna_sequence = f.read()
f.close()

# Tokenize the sequence. 
# Need to check if this is a valid tokenization. Maybe add/remove parameters 
tokens = tokenizer.encode_plus(
    dna_sequence,
    padding="max_length",
    truncation=True,
    is_split_into_words=True,
    return_tensors="pt"  # pt for pytorch
)

# Extract the input tensors
input_ids = tokens["input_ids"]

# EMBEDDINGS:
# Set BertModel pretrained dnabert model:
model = BertModel.from_pretrained(model_name)

# Get BERT embeddings
with torch.no_grad(): # torch.no_grad() tells PyTorch to not calculate the gradients (we don't want to interupt the model's gratient track) 
    outputs = model(input_ids)

# Get the last layer hidden states (DNABERT embeddings)
hidden_states = outputs.last_hidden_state
print(hidden_states)
print(len(hidden_states[0]))
print(len(hidden_states[0][0]))



# CLASSIFICATION:
    # Questions: 
    # How to set classes? 
    # Should we train the model first?
#Basic Classification setup:

# Set BertForSequenceClassification pretrained dnabert model:
classifier_model = BertForSequenceClassification.from_pretrained(model_name)

# Make predictions
with torch.no_grad(): # torch.no_grad() tells PyTorch to not calculate the gradients
    outputs = classifier_model(input_ids)

# Get the predicted probabilities
probs = torch.softmax(outputs.logits, dim=1)[0]  # logits(p) = log(p/(1-p))
print(probs)
loss = outputs.loss # need to train to get something here. for now, should be None


# NOTES
# As we run the code we got this notification (need to check it):

"""Some weights of the model checkpoint at zhihan1996/DNA_bert_6 were not used when initializing BertModel: ['cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.decoder.bias']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of the model checkpoint at zhihan1996/DNA_bert_6 were not used when initializing BertForSequenceClassification: ['cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.decoder.bias']
- This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at zhihan1996/DNA_bert_6 and are newly initialized: ['classifier.weight', 'classifier.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."""

