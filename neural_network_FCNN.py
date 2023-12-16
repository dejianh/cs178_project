from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sample_size = 10000
ngram_range=(1, 2) # combination of words
max_features = 1000
hidden_sizes = (100, )
learning_rate = 0.001
epochs = 5 # number of passes through dataset


class ReviewRatingPredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=5):
        super(ReviewRatingPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_sizes[0], output_size)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
    
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

print(combined_data['stars'].value_counts())
print('Combine Data Complete')

vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features, ngram_range=ngram_range)
X = vectorizer.fit_transform(combined_data['text']).toarray()
y = combined_data['stars'].values - 1  # Adjusting labels to 0-4

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Utilize Dataset to save memory
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 500 # larger batch size for parallel processing for gpu

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

hidden_layer_sizes = [2, 4, 6, 8, 10, 25, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]

train_error_rates = []
test_error_rates = []

for h1 in (hidden_layer_sizes):
    hidden_sizes = (h1,)

    input_size = max_features
    output_size = 5  # number of classes or in this case stars

    # Create the model
    model = ReviewRatingPredictor(input_size, hidden_sizes, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("training begins")

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