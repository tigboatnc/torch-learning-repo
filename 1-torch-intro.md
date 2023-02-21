


- Progress https://pytorch.org/tutorials/beginner/basics/data_tutorial.html - custom dataloader 



# Torch 

```toc
```

## ML Pipeline 
A typical ML pipeline can be divided into the following steps :
	Data -> Model -> Training -> Inference 
- **Data** : EDA, Building dataloaders 
- **Model** : Designing models 
- **Training** : Designing the training loop 
- **Inference** : Inferring results 
- **Model Optimization (Optional)** : Optimizing model for inference 


### 1. Data 
- 2 main data primitives in torch : 
	1. DataLoader `torch.utils.data.DataLoader`
		1. Takes the dataset as input and converts it into an iterable
	2. Dataset `torch.utils.data.Dataset`
		1. Contains the dataset samples

#### Basic Implementation
Imports
```python 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
```


Creating a dataset instance 
```python
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```

Creating a dataloader 
```python
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
```

#### Custom Dataloaders




### 2. Model 
#### Model Definition
- All models inherit from the `torch.nn` module 
```python
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```

#### Optimizer
```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```


### 3. Training Loop 
```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

```python
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

### 4. Inference 
####  Model Operations 
saving models
```python
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
```

loading models
```python
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
```

inference 
```python
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

------- 

## Tensors
- Tensors are the core data structures of ML models. 
- Tensors encode the input and output of NNs. 
- Tensors are optimized for GPU acceleration 
- Very similar to `numpy` and in many cases tensor objects share the same memory

### Properties of tensors 
- `shape` : `tensor.shape`
	- `torch.Size([3, 4])`
- `dtype` : `torch.dtype`
	- All tensors are associated with a shape 
	- eg. `torch.float32`
- `device` : `torch.device`
	- All tensors are also stored on a memory (gpu,cpu etc)
	- eg. `cpu`

### Tensor Initialization
```python
# init
tensorVar = torch.tensor([[20,10]])

# from a numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# from another tensor 
x_ones = torch.ones_like(x_data) # retains the properties of x_data
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data

# initializing from shape 
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

```

### Tensor Operations 

- A lot of tensor operations need to specify an output, and these functions come with an optional parameter which can be set using `_` ie. `add_` instead of `add`

- Indexing 
	- Can use numpy like indexing 
	- Row : `tensorVar[0]`
	- First Column : `tensorVar[:,0]`
	- Last Column : `tensorVar[...,-1]`
- Concatenate 
	- `tensor.cat([t1,t2,t3],dim=1)`