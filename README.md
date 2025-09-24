# Deep-Learning-workshop
## NAME: SARWESHVARAN A
## REG NO: 212223230198
## Aim
Build and train a deep learning model using embeddings for categorical data and batch normalization for continuous data to classify income levels.

## Input Features
- **Categorical:** sex, education, marital-status, workclass, occupation  
- **Continuous:** age, hours-per-week  
- **Target:** label (income class)

## Theory  
Use embedding layers to represent categorical variables, batch normalization for continuous variables, and fully connected layers with dropout and ReLU activations to build a robust classifier.

## Procedure  
1. Load and inspect dataset (`income.csv`).  
2. Convert categorical columns to category type and shuffle data.  
3. Create embedding sizes and convert data to tensors.  
4. Split data into training and test sets.  
5. Define the PyTorch tabular model with embeddings and batch normalization.  
6. Initialize model, loss function (CrossEntropyLoss), and Adam optimizer.  
7. Train the model for 300 epochs, printing loss every 25 epochs.  
8. Plot training loss curve to evaluate learning.  
9. Optionally, experiment with model architecture and hyperparameters.

## Prerequisites  
- Python 3.x, PyTorch, pandas, matplotlib, scikit-learn installed.  
- Basic Python and ML knowledge.

## Program
```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

df = pd.read_csv('income.csv')

print(f"Length of the dataset: {len(df)}")
print("First 5 rows of the dataset:")
print(df.head())
```
<img width="479" height="219" alt="image" src="https://github.com/user-attachments/assets/68c24b92-2267-4fd3-b888-dc5607646f44" />

```python
cat_cols = ['sex', 'education', 'marital-status', 'workclass', 'occupation']
cont_cols = ['age', 'hours-per-week']
y_col = ['label']

print(f'cat_cols has {len(cat_cols)} columns')
print(f'cont_cols has {len(cont_cols)} columns')
print(f'y_col has {len(y_col)} column')
```
<img width="175" height="51" alt="image" src="https://github.com/user-attachments/assets/2288fed7-2a62-43fd-8847-0d10574fd836" />

```python
for col in cat_cols: # Convert categorical columns to category type
    df[col] = df[col].astype('category')

# Shuffle the data
df = shuffle(df, random_state=101)
df.reset_index(drop=True, inplace=True)

print(df.head()) # Preview the first 5 rows of the shuffled dataset
```

<img width="475" height="198" alt="image" src="https://github.com/user-attachments/assets/c76bd875-21c3-46c6-9595-ad73656efe43" />

```python
# Create embedding sizes for each categorical column
cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
print(f"Embedding sizes: {emb_szs}")

# Convert categorical data to numeric codes
cats = torch.tensor(np.stack([df[col].cat.codes.values for col in cat_cols], axis=1), dtype=torch.long)

# Convert continuous data to tensors
conts = torch.tensor(np.stack([df[col].values for col in cont_cols], axis=1), dtype=torch.float32)

y = torch.tensor(df[y_col].values).flatten() # Convert labels to tensor

# Split data into training and test sets
b = 30000  # batch size
t = 5000   # test size

cat_train = cats[:b-t]
cat_test  = cats[b-t:b]
con_train = conts[:b-t]
con_test  = conts[b-t:b]
y_train   = y[:b-t]
y_test    = y[b-t:b]

print(f"cat_train shape: {cat_train.shape}")
print(f"con_train shape: {con_train.shape}")
print(f"y_train shape: {y_train.shape}")
```
<img width="391" height="74" alt="image" src="https://github.com/user-attachments/assets/faf43eda-9b3e-43d8-8b0b-b1218070d2e4" />
 
```python
class TabularModel(nn.Module):
    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        
        # Create embedding layers for categorical columns
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        
        # Dropout for embeddings and batch normalization for continuous columns
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)

        # Build hidden layers
        layerlist = []
        n_emb = sum((nf for ni, nf in emb_szs))
        n_in = n_emb + n_cont

        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1], out_sz))
        
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cat, x_cont):
        # Process categorical data through embedding layers
        embeddings = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)

        # Process continuous data through batch normalization
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)

        # Pass through the hidden layers
        x = self.layers(x)
        return x
# Initialize the model, loss function, and optimizer
torch.manual_seed(33)
model = TabularModel(emb_szs, n_cont=len(cont_cols), out_sz=2, layers=[50], p=0.4)

# Use CrossEntropyLoss for classification and Adam optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)
```
<img width="640" height="264" alt="image" src="https://github.com/user-attachments/assets/451bd7c7-3cd7-40c7-905b-c4fcd3ab339c" />

```python
# Training loop
epochs = 300
losses = []

for i in range(1, epochs + 1):
    # Forward pass
    y_pred = model(cat_train, con_train)
    
    # Compute loss
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())

    # Print loss every 25 epochs
    if i % 25 == 1:
        print(f'Epoch {i:3} Loss: {loss.item():.8f}')
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot the loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training Loss Curve')
plt.show()
```
<img width="562" height="563" alt="image" src="https://github.com/user-attachments/assets/b966ae11-7faa-46a0-ae9a-fc489c5c1916" />

```python
with torch.no_grad():
    y_val = model(cat_test, con_test)
    loss = criterion(y_val, y_test)
print(f'CE Loss: {loss:.8f}')
```

<img width="126" height="26" alt="image" src="https://github.com/user-attachments/assets/04c05136-97e7-48fc-93b8-8c1de86fcb56" />


```python
correct = 0
for i in range(len(y_test)):
    if y_val[i].argmax().item() == y_test[i].item():
        correct += 1

accuracy = correct / len(y_test) * 100
print(f'{correct} out of {len(y_test)} = {accuracy:.2f}% correct')
```

<img width="227" height="21" alt="image" src="https://github.com/user-attachments/assets/cf0a6991-fef8-4c79-952e-d9f6fc6b3be5" />








