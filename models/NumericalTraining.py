import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(1, 20000)
        self.output = nn.Linear(20000, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        return self.output(x)

def save_model(model, optimizer, scheduler, filename="model.pth"):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, filename)
    print("✅ Model saved successfully.")

def load_model(model, optimizer, scheduler, filename="model.pth"):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✅ Model loaded from {filename}. Resuming training.")
    else:
        print("🚀 No saved model found. Starting new training.")

x_train = torch.tensor([
    [10.0], [20.0], [15.0], [25.0], [30.0],
    [40.0], [50.0], [60.0], [70.0], [80.0]
], dtype=torch.float32)

y_train = torch.tensor([
    [40.0], [65.0], [52.5], [77.5], [90.0],
    [115.0], [140.0], [165.0], [190.0], [215.0]
], dtype=torch.float32)

model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)

load_model(model, optimizer, scheduler)

def train_model(epochs):
    print("⏳ Training model...")
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, LR: {scheduler.get_last_lr()}")

        if epoch % 1000 == 0:
            save_model(model, optimizer, scheduler)

        if epochs_no_improve >= 1000:
            print("🚨 Early stopping: No improvement in 1000 epochs.")
            break

def use_model():
    num = float(input("Enter a number: "))
    input_tensor = torch.tensor([[num]], dtype=torch.float32)
    with torch.no_grad():  
        predicted_output = model(input_tensor).item()
    print(f"Model prediction: {predicted_output:.2f}")


def menu():
    while True:
        print("\n--- Menu ---")
        print("1. Train model")
        print("2. Use model to predict")
        print("3. Exit")
        option = input("Select an option: ")

        if option == "1":
            epochs = int(input("Enter the number of epochs to train: "))
            train_model(epochs)
        elif option == "2":
            use_model()
        elif option == "3":
            print("Exiting the program...")
            break
        else:
            print("Invalid option. Please select a valid option.")

menu()
