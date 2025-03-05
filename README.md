# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

It consists of an input layer with 1 neuron, two hidden layers with 4 neurons each, and an output layer with 1 neuron. Each neuron in one layer is connected to all neurons in the next layer, allowing the model to learn complex patterns. The hidden layers use activation functions such as ReLU to introduce non-linearity, enabling the network to capture intricate relationships within the data. 
During training, the model adjusts its weights and biases using optimization techniques like RMSprop or Adam, minimizing a loss function such as Mean Squared Error for regression.The forward propagation process involves computing weighted sums, applying activation functions, and passing the transformed data through layer.

## Neural Network Model

![image](https://github.com/user-attachments/assets/0a06c931-28a8-4829-8fed-d4ef0f71f620)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: YOGESHVAR M
### Register Number: 212222230180
```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
```python
ai_brain = NeuralNet ()
criterion = nn. MSELoss ()
optimizer = optim.RMSprop (ai_brain. parameters(), lr=0.001)
```

```python
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=4000) :
  for epoch in range (epochs) :
    optimizer. zero_grad()
    loss = criterion(ai_brain(X_train), y_train)
    loss. backward()
    optimizer.step()
    ai_brain. history['loss'] .append(loss.item())
    if epoch % 200 == 0:
      print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```

## Dataset Information

![image](https://github.com/user-attachments/assets/b560342c-3a35-47ad-812f-29808c6959ab)

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/101c949a-d33b-4780-be31-41634e6fb8e3)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/ffdc1b0d-d1cc-4199-9226-7ca81d4c28b8)

## RESULT

Thus a neural network regression model is developed successfully.The model demonstrated strong predictive performance on unseen data, with a low error rate.
