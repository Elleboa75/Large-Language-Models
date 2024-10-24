import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load the tokenizer and the pre-trained BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # 3 classes: antiracist, neutral, racist

# Load the processed data
anti_df = pd.read_csv('../Data/processed/antiracist_data.csv')
neutral_df = pd.read_csv('../Data/processed/neutral_data.csv')
racist_df = pd.read_csv('../Data/processed/racist_data.csv')

# Combine datasets and create labels
anti_df['label'] = 0  # Label for antiracist
neutral_df['label'] = 1  # Label for neutral
racist_df['label'] = 2  # Label for racist

# Combine all data into a single DataFrame
df = pd.concat([anti_df, neutral_df, racist_df], ignore_index=True)

# Shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

# Tokenization and encoding of the dataset
def tokenize_data(sentences, tokenizer, max_length=128):
    encoded_inputs = tokenizer(
        sentences, 
        return_tensors='pt', 
        truncation=True, 
        padding=True, 
        max_length=max_length
    )
    return encoded_inputs['input_ids'], encoded_inputs['attention_mask']

# Tokenize the text data
input_ids, attention_masks = tokenize_data(df['text'].tolist(), tokenizer)

# Encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['label'])

# Split the data into training and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.1)
train_masks, val_masks = train_test_split(attention_masks, test_size=0.1)

# Convert data to torch tensors
train_inputs = train_inputs
val_inputs = val_inputs
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)
train_masks = train_masks
val_masks = val_masks

# Create DataLoader for training and validation
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=16)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_dataloader = DataLoader(val_data, batch_size=16)

# Optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
epochs = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        batch_inputs = batch[0].to(device)
        batch_masks = batch[1].to(device)
        batch_labels = batch[2].to(device)

        model.zero_grad()

        outputs = model(input_ids=batch_inputs, attention_mask=batch_masks, labels=batch_labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}')

# Validation loop
model.eval()
val_predictions = []
val_labels_list = []

with torch.no_grad():
    for batch in val_dataloader:
        batch_inputs = batch[0].to(device)
        batch_masks = batch[1].to(device)
        batch_labels = batch[2].to(device)

        outputs = model(input_ids=batch_inputs, attention_mask=batch_masks)
        logits = outputs.logits

        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        val_predictions.extend(predictions)
        val_labels_list.extend(batch_labels.cpu().numpy())

# Accuracy and classification report
accuracy = accuracy_score(val_labels_list, val_predictions)
report = classification_report(val_labels_list, val_predictions, target_names=label_encoder.classes_)

print(f'Validation Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
