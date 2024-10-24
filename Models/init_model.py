import torch
import os
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from tqdm import tqdm

# File paths to save/load preprocessed data
preprocessed_data_path = '../Data/processed/'
tokenized_inputs_file = os.path.join(preprocessed_data_path, 'input_ids.pt')
tokenized_masks_file = os.path.join(preprocessed_data_path, 'attention_masks.pt')
labels_file = os.path.join(preprocessed_data_path, 'labels.pt')

# Load the tokenizer and the pre-trained BERT model
print("Loading BERT tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # 3 classes: antiracist, neutral, racist
print("Model and tokenizer loaded successfully!")

# Check if preprocessed data exists
if os.path.exists(tokenized_inputs_file) and os.path.exists(tokenized_masks_file) and os.path.exists(labels_file):
    print("Preprocessed data found. Loading from disk...")
    input_ids = torch.load(tokenized_inputs_file)
    attention_masks = torch.load(tokenized_masks_file)
    labels = torch.load(labels_file)
    print("Preprocessed data loaded successfully!")
else:
    print("Preprocessed data not found. Starting preprocessing...")

    # Load the processed data
    print("Loading processed data...")
    anti_df = pd.read_csv('../Data/processed/antiracist_data.csv')
    neutral_df = pd.read_csv('../Data/processed/neutral_data.csv')
    racist_df = pd.read_csv('../Data/processed/racist_data.csv')

    # Combine datasets and create labels
    print("Combining datasets...")
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

    print("Tokenizing text data...")
    input_ids, attention_masks = tokenize_data(df['text'].tolist(), tokenizer)

    # Encode the labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    labels = torch.tensor(label_encoder.fit_transform(df['label']))

    # Save the tokenized inputs, attention masks, and labels to disk for future use
    print("Saving preprocessed data to disk...")
    torch.save(input_ids, tokenized_inputs_file)
    torch.save(attention_masks, tokenized_masks_file)
    torch.save(labels, labels_file)
    print("Preprocessed data saved successfully!")

# Split the data into training and validation sets
print("Splitting data into training and validation sets...")
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.1)
train_masks, val_masks = train_test_split(attention_masks, test_size=0.1)

# Create DataLoader for training and validation
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=16)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_dataloader = DataLoader(val_data, batch_size=16)

# Optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop with progress bar
epochs = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print("Starting training...")

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    model.train()
    total_loss = 0
    
    # tqdm progress bar for training
    progress_bar = tqdm(train_dataloader, desc="Training", leave=False)
    
    for step, batch in enumerate(progress_bar):
        batch_inputs = batch[0].to(device)
        batch_masks = batch[1].to(device)
        batch_labels = batch[2].to(device)

        model.zero_grad()

        outputs = model(input_ids=batch_inputs, attention_mask=batch_masks, labels=batch_labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        # Update progress bar with current loss
        progress_bar.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

# Validation loop with progress bar
print("\nStarting validation...")

model.eval()
val_predictions = []
val_labels_list = []

# tqdm progress bar for validation
progress_bar = tqdm(val_dataloader, desc="Validation", leave=False)

with torch.no_grad():
    for batch in progress_bar:
        batch_inputs = batch[0].to(device)
        batch_masks = batch[1].to(device)
        batch_labels = batch[2].to(device)

        outputs = model(input_ids=batch_inputs, attention_mask=batch_masks)
        logits = outputs.logits

        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        val_predictions.extend(predictions)
        val_labels_list.extend(batch_labels.cpu().numpy())

# Calculate accuracy and classification report
accuracy = accuracy_score(val_labels_list, val_predictions)
report = classification_report(val_labels_list, val_predictions, target_names=label_encoder.classes_)

print(f"\nValidation Accuracy: {accuracy:.4f}")
print(f"\nClassification Report:\n{report}")
