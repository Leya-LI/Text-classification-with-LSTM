import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from preprocess_data import preprocess_data
from LSTM import SentimentLSTM
from utils import calculate_accuracy, plot_training_history, save_model


def train_model(
        csv_path,
        batch_size=64,
        learning_rate=0.001,
        num_epochs=10,
        embedding_dim=100,
        hidden_dim=256,
        n_layers=2,
        dropout=0.5
):
    # Preprocess data
    train_dataset, test_dataset, vocab, label_encoder = preprocess_data(csv_path)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = SentimentLSTM(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=len(set(label_encoder)),
        n_layers=n_layers,
        dropout=dropout
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training history tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for batch in train_loader:
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(texts)
            loss = criterion(predictions, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Tracking metrics
            train_loss += loss.item()
            train_correct += (predictions.argmax(1) == labels).float().sum().item()
            train_total += labels.size(0)

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for batch in test_loader:
                texts = batch['text'].to(device)
                labels = batch['label'].to(device)

                predictions = model(texts)
                loss = criterion(predictions, labels)

                val_loss += loss.item()
                val_correct += (predictions.argmax(1) == labels).float().sum().item()
                val_total += labels.size(0)

        # Calculate metrics
        train_epoch_loss = train_loss / len(train_loader)
        train_epoch_acc = train_correct / train_total * 100

        val_epoch_loss = val_loss / len(test_loader)
        val_epoch_acc = val_correct / val_total * 100

        # Store history
        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)
        train_accs.append(train_epoch_acc)
        val_accs.append(val_epoch_acc)

        # Print epoch results
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.2f}%')
        print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%')

    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)

    # Save the model
    save_model(model, 'sentiment_lstm_model.pth')

    return model, vocab, label_encoder


# Main execution
if __name__ == '__main__':
    train_model('./data/train.tsv')
