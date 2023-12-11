from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

array_split = 1000000
ngram_range=(1, 2) # combination of words
max_features = 10000
hidden_size = 500
learning_rate = 0.007
epochs = 15 # number of passes through dataset


class ReviewRatingPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=5):
        super(ReviewRatingPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output layer has 5 neurons for 5 classes

    def forward(self, x):
        # Apply ReLU activation function after the first layer
        x = F.relu(self.fc1(x))
        # Output layer
        x = self.fc2(x)
        return x
    

data = pd.read_json('yelp_dataset/yelp_academic_dataset_review.json', lines=True)

star1_comments = data[data['stars'] == 1.0]['text'].iloc[:array_split].tolist()
star2_comments = data[data['stars'] == 2.0]['text'].iloc[:array_split].tolist()
star3_comments = data[data['stars'] == 3.0]['text'].iloc[:array_split].tolist()
star4_comments = data[data['stars'] == 4.0]['text'].iloc[:array_split].tolist()
star5_comments = data[data['stars'] == 5.0]['text'].iloc[:array_split].tolist()


df_star1 = pd.DataFrame({'text': star1_comments, 'stars': 1})
df_star2 = pd.DataFrame({'text': star2_comments, 'stars': 2})
df_star3 = pd.DataFrame({'text': star3_comments, 'stars': 3})
df_star4 = pd.DataFrame({'text': star4_comments, 'stars': 4})
df_star5 = pd.DataFrame({'text': star5_comments, 'stars': 5})

combined_data = pd.concat([df_star1, df_star2, df_star3, df_star4, df_star5], ignore_index=True)
comments = combined_data['text']

vectorizer = CountVectorizer(stop_words='english', max_features=max_features, ngram_range=ngram_range)
bow_features = vectorizer.fit_transform(comments)

# dense array to use in pytorch
bow_array = bow_features.toarray()

labels = combined_data['stars'].values

# Splitting the first half for training and the second half for testing
train_size = len(bow_array) // 2

X_train = bow_array[:train_size]
y_train = labels[:train_size]

X_test = bow_array[train_size:]
y_test = labels[train_size:]

# Adjust labels for classification (0 to 4 for 1 to 5 stars)
y_train_indices = y_train - 1
y_test_indices = y_test - 1

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train_indices, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_indices, dtype=torch.long)


input_size = max_features
hidden_size = hidden_size
output_size = 5  # number of classes or in this case stars

# Create the model
model = ReviewRatingPredictor(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(epochs):
    model.train()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


model.eval()

with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    test_loss = criterion(y_pred_test, y_test_tensor)
    print(f'Test Loss: {test_loss.item()}')


test_comment = "This has grown to be one of my favorite stores in all of Indy. Supplements? Check? Awesome clean make-up and toiletries that I was spending a fortune buying online? Check? Other organic goodies that I've read about online and haven't found anywhere else until now? Check! Local, local, local? Check!! I mean seriously I have yet to call them to ask if they have such and such and have them say no. Coconut flour? Yes. CoQ10 face wash? Yes. Great Lakes grass-fed gelatin? Yes!\n\nI love you GENF! Can I move in?\n\nProducts: 5\/5\nService: 5\/5\nLocation: 5\/5"
test_vectorized = vectorizer.transform([test_comment])
test_comment_tensor = torch.tensor(test_vectorized.toarray(), dtype=torch.float32)

model.eval()
with torch.no_grad():
    predicted_logits = model(test_comment_tensor)
    predicted_class = torch.argmax(predicted_logits, dim=1)
    predicted_star = predicted_class + 1  # Adjusting to the 1-5 star scale
    print("Predicted Star Rating:", predicted_star.item())