import torch
import torch.nn.functional as F
import pandas as pd
from utils.metrics import calculate_metrics

def train_model(model, train_data, test_data, num_epochs=50, learning_rate=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    train_losses, test_losses = [], []

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(train_data)
        train_loss = F.nll_loss(out, train_data.y)
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            out_test = model(test_data)
            test_loss = F.nll_loss(out_test, test_data.y)
            test_losses.append(test_loss.item())
        model.train()

    loss_data = pd.DataFrame({"Epoch": range(1, num_epochs + 1), "Training Loss": train_losses, "Testing Loss": test_losses})
    return model, loss_data

def evaluate_model(model, test_data):
    model.eval()
    with torch.no_grad():
        out = model(test_data)
        pred = out.argmax(dim=1).detach().numpy()
        y_true = test_data.y.detach().numpy()
        metrics = calculate_metrics(y_true, pred)
    return metrics
