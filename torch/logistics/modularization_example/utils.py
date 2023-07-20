
"""
Utility functions to save and load PyTorch models.
"""
import torch
import pathlib

def get_device():
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    
    return device

def accuracy_fn(y_true, y_pred):
    return (y_pred == y_true).sum().item()/len(y_pred)

def save_model(
    model: torch.nn.Module,
    target_dir_path: pathlib.PosixPath,
    filename: str
):
    """Saves a PyTorch model to a target directory.

    Args:
        model: PyTorch Model to save.
        target_dir_path: Target directory for the saved model.
        filename: A filename for the saved model. It should include
            ".pth" or ".pt" file extensions.

    Example usage:
        save_model(
            model=model,
            target_dir_path="models",
            filename="model.pth"
        )
    """
    # 1. Create the target directory
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert filename.endswith(".pth") or filename.endswith(".pt"), "filename should end with '.pt' or '.pth'"
    model_path = target_dir_path / filename

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_path}")
    torch.save(obj=model.state_dict(), f=model_path)
