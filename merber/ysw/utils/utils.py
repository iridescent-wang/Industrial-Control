import pandas as pd
import numpy as np
import random
import torch
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PcapDataset(Dataset):

    def __init__(self, features, labels):
        super().__init__()
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i], self.labels[i]


def get_data_loader(batch_size=128, train=True):
    DIR = 'dataset/'
    df = pd.read_csv(DIR + ('train' if train else 'test') + '.csv')
    features = [normalization(torch.tensor(data)) for *data, _ in df.itertuples(index=False)]
    labels = list(df['Label'].values)
    return DataLoader(PcapDataset(features, labels), batch_size=batch_size, shuffle=True)


def normalization(x):
    return (x - torch.min(x)) / (torch.max(x) - torch.min(x))


def categorical_accuracy(preds, y):
    top_pred = preds.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for features, labels in iterator:
        features = features.to('cuda')
        labels = labels.to('cuda').long()

        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, labels)

        acc = categorical_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for features, labels in iterator:
            features = features.to('cuda')
            labels = labels.to('cuda').long()

            predictions = model(features)
            loss = criterion(predictions, labels)
            acc = categorical_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def _test(model, iterator):
    model.eval()
    y, y_pred = [], []

    with torch.no_grad():
        for features, labels in iterator:
            features = features.to('cuda')
            predictions = model(features).argmax(dim=1)
            y.extend(labels.tolist())
            y_pred.extend(predictions.tolist())

    return y, y_pred


def testing(model, iterator, labels):
    y, y_pred = _test(model, iterator)

    print('Accuracy:', metrics.accuracy_score(y, y_pred))
    print(metrics.classification_report(y, y_pred, target_names=labels, digits=5))
