# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

It consists of an input layer with 1 neuron, two hidden layers with 8 neurons in the first layer and 10 neurons in the second layer, and an output layer with 1 neuron. Each neuron in one layer is connected to all neurons in the next layer, allowing the model to learn complex patterns. The hidden layers use activation functions such as ReLU to introduce non-linearity, enabling the network to capture intricate relationships within the data. 
During training, the model adjusts its weights and biases using optimization techniques like RMSprop or Adam, minimizing a loss function such as Mean Squared Error for regression.The forward propagation process involves computing weighted sums, applying activation functions, and passing the transformed data through layer.

## Neural Network Model

![image](https://github.com/user-attachments/assets/4e51783a-39ef-4737-a58d-3a58d1aa5caf)

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

![image](https://github.com/user-attachments/assets/821c2331-6983-4538-ab4b-9265340f8c1e)

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/c783cb1e-df4e-4182-8ba3-522854921f7c)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/52fe09cf-5bf9-4754-83ed-605d5730ce8b)

## RESULT

Thus a neural network regression model is developed successfully.The model demonstrated strong predictive performance on unseen data, with a low error rate.
