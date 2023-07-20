
"""
Contains training functions.
"""

import torch

import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Callable

def train_step(model, dataloader, loss_fn, optimizer, accuracy_fn, device):
    ### Setup initial training state
    training_loss, training_accuracy = 0., 0.
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        # Send data to the appropriate device
        X, y = X.to(device), y.to(device)

        # 1. Forward Pass
        y_logits = model(X)
        y_pred = y_logits.softmax(1).argmax(1)

        # 2. Compute Loss and Accuracy
        loss = loss_fn(y_logits, y)
        accuracy = accuracy_fn(y, y_pred)

        training_loss += loss.item()
        training_accuracy += accuracy

        # 3. Clear Optimizer Gradients
        optimizer.zero_grad()

        # 4. Backward Pass
        loss.backward()

        # 5. Update Weights
        optimizer.step()
    
    ### Aggregate losses and accuracies
    training_loss /= len(dataloader)
    training_accuracy /= len(dataloader)

    return training_loss, training_accuracy

def test_step(model, dataloader, loss_fn, accuracy_fn, device):
    ### Setup epoch evaluation state
    test_loss, test_accuracy = 0., 0.
    model.eval()

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            # Send data to the appropriate device
            X, y = X.to(device), y.to(device)

            # 1. Forward Pass
            y_logits = model(X)
            y_pred = y_logits.softmax(1).argmax(1)

            # 2. Compute Loss and Accuracy
            loss = loss_fn(y_logits, y)
            accuracy = accuracy_fn(y, y_pred)

            test_loss += loss.item()
            test_accuracy += accuracy
    
        ### Aggregate losses and accuracies for the test set.
        test_loss /= len(dataloader)
        test_accuracy /= len(dataloader)
    
    return test_loss, test_accuracy

def train(
    model: torch.nn.Module, 
    train_dataloader: DataLoader, 
    test_dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer,
    accuracy_fn: Callable,
    loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
    device: str = 'cpu',
    epochs: int = 5
):
    
    # 1. Create a container for the results
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # 2. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device
        )
        
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device
        )
        
        # 3. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 4. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 5. Return the filled results at the end of the epochs
    return results
