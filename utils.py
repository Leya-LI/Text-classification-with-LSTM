import torch
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def calculate_accuracy(preds, labels):
    """
    Calculate accuracy of predictions

    Args:
        preds (torch.Tensor): Model predictions
        labels (torch.Tensor): True labels

    Returns:
        float: Accuracy percentage
    """
    pred_classes = torch.argmax(preds, dim=1)
    correct = (pred_classes == labels).float().sum()
    return (correct / len(labels)) * 100


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """
    Plot training and validation loss and accuracy

    Args:
        train_losses (list): Training loss for each epoch
        val_losses (list): Validation loss for each epoch
        train_accs (list): Training accuracy for each epoch
        val_accs (list): Validation accuracy for each epoch
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy plot
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def save_model(model, path):
    """
    Save model state

    Args:
        model (torch.nn.Module): Model to save
        path (str): File path to save model
    """
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """
    Load model state

    Args:
        model (torch.nn.Module): Model to load state into
        path (str): File path to load model from

    Returns:
        torch.nn.Module: Model with loaded state
    """
    model.load_state_dict(torch.load(path))
    return model
