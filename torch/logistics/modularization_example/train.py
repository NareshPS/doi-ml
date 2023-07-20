
import os
import torch
import data_setup, engine, module_builder, utils

import torch.nn as nn

from pathlib import Path
from torchvision import transforms
from timeit import default_timer as timer

# 1. Setup hyperparameters
BATCH_SIZE, EPOCHS, LEARNING_RATE = 32, 5, 0.001
HIDDEN_UNITS = 10
IMG_SIZE = 64

def main():
    # 2. Setup path to data folder
    data_path = Path(os.path.expanduser("~/.datasets/pizza_steak_sushi"))
    train_path = data_path / 'train'
    test_path = data_path / 'test'

    # 3. Setup target device
    device = utils.get_device()

    # 4. Create data transforms
    data_transform = transforms.Compose([
        transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    # 5. Create dataloaders
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_path=train_path,
        test_path=test_path,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    # 6. Setup a random seed
    torch.manual_seed(SEED)
    torch.mps.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # 7. Create a model
    model = module_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    print('Model Architecture')
    print('------------------')
    print(f'{model} on {device}')

    print('\nModel Parameters')
    print(f'-------------------')
    for name in model.state_dict().keys():
        print(f'{name}')

    # for (name, param) in model_0.state_dict().items():
    #     print(f'{name}: {param}')

    xt = torch.randn(2, 3, IMG_SIZE, IMG_SIZE, device=device)
    yt = model(xt)

    print('\nInputs and Outputs')
    print('-------------------')
    print(f'Input: {xt}')
    print(f'Output: {yt}')
    print(f'Input Shape: {xt.shape} Output Shape: {yt.shape}')

    # 8. Setup loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=LEARNING_RATE, params=model.parameters())

    # 9. Train the model
    print('Training Loop')
    print('-------------')
    print('-------------')

    start_time = timer()

    model_results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        accuracy_fn=utils.accuracy_fn,
        epochs=EPOCHS,
        device=device
    )

    end_time = timer()

    print(f"Total training time: {end_time - start_time:.3f} seconds")

    # 10. Save the model
    utils.save_model(
        model=model,
        target_dir_path=Path('models'),
        filename='food_classification.pth'
    )

if __name__ == '__main__':
    main()
