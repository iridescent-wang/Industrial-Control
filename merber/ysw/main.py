import time
import torch
import torch.nn as nn
import torch.optim as optim
from utils.preprocessing import preprocessing
from utils.utils import set_seed, get_data_loader, epoch_time, train, evaluate, testing
from model.classifier import MLP

# parameters
IN_DIM = 76
OUT_DIM = 12
LR = 0.001
WEIGHT_DECAY = 1e-6
N_EPOCHS = 100
BATCH_SIZE = 64
DEVICE = 'cuda'


def main():
    labels = preprocessing()

    set_seed()

    model = MLP(IN_DIM, OUT_DIM).to(DEVICE)

    train_iterator = get_data_loader(batch_size=BATCH_SIZE, train=True)
    valid_iterator = get_data_loader(batch_size=BATCH_SIZE, train=False)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_model/model.pt')

        # print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        # print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        # print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    model.load_state_dict(torch.load('saved_model/model.pt'))
    testing(model, valid_iterator, labels)


if __name__ == '__main__':
    main()
