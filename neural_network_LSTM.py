import pandas as pd
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

sample_size = 10000
learning_rate = 0.005
max_length = 100
embedding_dim = 500
hidden_dim = 250
output_dim = 5
stacked_layers = 2
dropout_rate = 0.25

epochs = 5 # number of passes through dataset


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_dim, num_layers, dropout_rate):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        hidden = lstm_out[:, -1, :]
        out = self.fc(hidden)
        return out

def combine_data():

    star1 = pd.read_parquet('output1.parquet')
    star2 = pd.read_parquet('output2.parquet')
    star3 = pd.read_parquet('output3.parquet')
    star4 = pd.read_parquet('output4.parquet')
    star5 = pd.read_parquet('output5.parquet')

    star1_comments = star1['text'].sample(n=sample_size, random_state=1)
    star2_comments = star2['text'].sample(n=sample_size, random_state=1)
    star3_comments = star3['text'].sample(n=sample_size, random_state=1)
    star4_comments = star4['text'].sample(n=sample_size, random_state=1)
    star5_comments = star5['text'].sample(n=sample_size, random_state=1)

    combined_data = pd.concat([
        pd.DataFrame({'text': star1_comments, 'stars': 1}),
        pd.DataFrame({'text': star2_comments, 'stars': 2}),
        pd.DataFrame({'text': star3_comments, 'stars': 3}),
        pd.DataFrame({'text': star4_comments, 'stars': 4}),
        pd.DataFrame({'text': star5_comments, 'stars': 5})
    ], ignore_index=True)

    return combined_data


# Testing custom tokenizer, does not work as well as torchtext built in tokenizer.
def tokenizers(text):
    # Tokenize and convert to lower case
    tokens = word_tokenize(text.lower())
    filtered_tokens = []

    for token in tokens:
        # Removing stopwords and punctuation
        if token.isalpha() and token not in stop_words:
            # Lemmatize tokens
            lemma = lemmatizer.lemmatize(token)
            filtered_tokens.append(lemma)

    return filtered_tokens

# padding to ensure same length for all comments
def pad_sequences(sequences, maxlen=max_length):
    padded_sequences = torch.zeros(len(sequences), maxlen, dtype=torch.long)
    for i, seq in enumerate(sequences):
        end = min(maxlen, len(seq))
        padded_sequences[i, :end] = seq[:end]
    return padded_sequences



combined_data = combine_data()
print('Combine Data Complete')


# Tokenize and process text
tokenizer = get_tokenizer('basic_english',)
tokenized = [tokenizer(comment) for comment in combined_data['text']]
print('Tokenized comments')

counter = Counter()
for tokens in tokenized:
    counter.update(tokens)

vocabs = vocab(counter, min_freq=1)

# Convert comments to integer sequences
encoded_comments = [torch.tensor([vocabs[token] for token in tokens]) for tokens in tokenized]
print("comments encoded")

padded_comments = pad_sequences(encoded_comments, maxlen=max_length)
print("comments padded")

# Prepare labels
labels = torch.tensor(combined_data['stars'].values) - 1

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded_comments, labels, test_size=0.5, random_state=42)

batch_size = 500

# Datasets and Dataloaders
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)


hidden_layer_sizes = [2, 4, 6, 8, 10, 15, 20, 25, 35, 45, 65, 85, 100, 150, 250, 400, 800]

train_error_rates = []
test_error_rates = []

for h in hidden_layer_sizes:

    model = LSTMClassifier(vocab_size=len(vocabs), embedding_dim=embedding_dim, hidden_dim=h, output_dim=output_dim, num_layers=stacked_layers, dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



    print('Training Begins')
    train_error_rate = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0
        n = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
            n += 1
        
        train_error_rate = 1 - correct_train / total_train
        print(f'Epoch {epoch}, Loss: {total_loss/n}, Train error rate: {train_error_rate}')
    train_error_rates.append(train_error_rate)

    model.eval()
    predictions = []
    correct = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            predictions.extend(list(zip(predicted.cpu().numpy(), labels.cpu().numpy())))

    test_loss /= total
    error_rate = 1 - correct/total
    print(f'Test Loss: {test_loss}')
    print(f'Error Rate: {error_rate}')

    test_error_rates.append(error_rate)

print(hidden_layer_sizes)
print(train_error_rates)
print(test_error_rates)