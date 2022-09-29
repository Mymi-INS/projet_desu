import torch
from torch import nn
from sklearn.metrics import accuracy_score


def train_loop(cnn, dataloader, device, criterion, optimizer, train=True):
    """
    Define a training loop.

    Inputs:
    ------

    cnn
        Instance of the model being trained.
    dataloader: torch.DataLoader
        Dataloader containing the train or test dataset.
    device: str
        The device being used to train the network.
    criterion
        Train criterion.
    optimizer
        Optimizer used in training.
    train: bool | True
        Whether it is a train or validation loop.

    Outputs:
    -------
    running_loss: float
        Loss values for an epoch.
    running_acc: 
        Accuracy value for an epoch.
    """

    running_loss = 0.0
    running_acc = 0.0

    if train:
        cnn.train()
    else:
        cnn.eval()

    for data in dataloader:

        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.type(torch.LongTensor).to(device)

        if train:
            optimizer.zero_grad()
        outputs = cnn.forward(inputs)

        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        # Compute accuracy
        y_prev = torch.nn.functional.softmax(outputs, -1).argmax(-1).to("cpu")
        running_acc += accuracy_score(y_prev, labels.to("cpu"))
        running_loss += loss.item()

    norm = len(dataloader)

    return running_loss / norm, running_acc / norm


def train(
    cnn,
    trainloader,
    testloader,
    device="auto",
    epochs: int = 100,
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    return_scores=False,
    return_train_scores=False,
    verbose: bool = False,
):
    """
    Inputs:
    ------

    cnn
        Instance of the model being trained.
    trainloader: torch.DataLoader
        Dataloader containing the train dataset.
    testloader: torch.DataLoader
        Dataloader containing the test dataset.
    device: str | "auto"
        The device being used to train the network.
    criterion | nn.CrossEntropyLoss
        Train criterion.
    optimizer | torch.optim.Adam
        Optimizer used in training.
    """
    # Select device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device

    # Send model to device
    cnn.to(device)

    if verbose:
        print(f"Using device: {device}")

    # Instantiate criterion
    criterion = criterion()
    # Instantiate optimizer
    optimizer = optimizer(cnn.parameters())

    running_loss_train, running_acc_train = 0, 0
    running_loss_val, running_acc_val = 0, 0

    for epoch in range(epochs):

        running_loss_train, running_acc_train = train_loop(cnn,
                                                           trainloader,
                                                           device, criterion,
                                                           optimizer,
                                                           train=True
                                                           )
        running_loss_val, running_acc_val = train_loop(cnn,
                                                       testloader, device,
                                                       criterion, optimizer,
                                                       train=False
                                                       )
        if verbose:
            print(
                "Epoch {:3d}: train loss {:.5f} | acc. train {:.3f} | val. loss {:.5f} | acc. val {:.3f}".format(
                    epoch + 1,
                    running_loss_train,
                    running_acc_train,
                    running_loss_val,
                    running_acc_val,
                ),
                end="\r",
            )

    if return_scores:
        if return_train_scores:
            return (running_loss_train, running_acc_train,
                    running_loss_val, running_acc_val)
        return (running_loss_val, running_acc_val)
