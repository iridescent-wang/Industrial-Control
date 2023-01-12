import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, *args, **kwargs):
        super().__init__()
        hid1_dim = 128
        hid2_dim = 64
        self.fc = nn.Sequential(nn.Linear(in_dim, hid1_dim), nn.ReLU(), nn.BatchNorm1d(hid1_dim), nn.Dropout(0.2),
                                nn.Linear(hid1_dim, hid2_dim), nn.ReLU(), nn.BatchNorm1d(hid2_dim), nn.Dropout(0.5),
                                nn.Linear(hid2_dim, out_dim))

    def forward(self, features):
        return self.fc(features)


# class MLP(nn.Module):

#     def __init__(self, in_dim, out_dim, *args, **kwargs):
#         super().__init__()
#         hid1_dim = 256
#         hid2_dim = 128
#         hid3_dim = 64
#         self.fc = nn.Sequential(nn.Linear(in_dim, hid1_dim), nn.ReLU(), nn.BatchNorm1d(hid1_dim), nn.Dropout(0.1),
#                                 nn.Linear(hid1_dim, hid2_dim), nn.ReLU(), nn.BatchNorm1d(hid2_dim), nn.Dropout(0.3),
#                                 nn.Linear(hid2_dim, hid3_dim), nn.ReLU(), nn.BatchNorm1d(hid3_dim), nn.Dropout(0.5),
#                                 nn.Linear(hid3_dim, out_dim))

#     def forward(self, features):
#         return self.fc(features)