import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from first_assign.cnn_utils import load_dataset


class GestureDataSet(Dataset):
    def __init__(self, mode='train'):
        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
        if mode == 'train':
            self.X_orig = X_train_orig
            self.Y_orig = Y_train_orig
        else:
            self.X_orig = X_test_orig
            self.Y_orig = Y_test_orig

        self.X_orig = self.X_orig / 255

    def __len__(self):
        return self.X_orig.shape[0]

    def __getitem__(self, idx):
        label = self.Y_orig[0][idx]
        image = self.X_orig[idx]
        return image, label


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # W1 = parameters['W1']  # (4, 4, 3, 8)
        # W2 = parameters['W2']  # (2, 2, 8, 16)
        self.conv1 = torch.nn.Sequential(     # input (N, 3, 64, 64)
            torch.nn.Conv2d(3, 8, (4, 4), 1), # (N, 8, 61, 61)
            torch.nn.LeakyReLU(1.0/20),
            torch.nn.MaxPool2d(8, 8)          #  (N, 8, 7, 7)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, (2, 2), 1), # (N, 16, 4, 4)
            torch.nn.LeakyReLU(1.0/20),
            torch.nn.MaxPool2d(4, 4)           # (N, 16, 1, 1)
        )
        self.dens1 = torch.nn.Linear(16, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dens1(x)
        return x

MODE_FILE_PATH = 'models/model.pkl'

if __name__ == '__main__':

    data_set = GestureDataSet('train')
    data_loader = DataLoader(data_set, batch_size=200)

    writer = SummaryWriter(comment='-lre3')

    model = CNN().double()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    try:
        saved_models = torch.load('models/model.pkl')
        model = saved_models['model']
        global_step = saved_models['step']
        print("after load global_step is {}".format(global_step))
    except Exception as e:
        global_step = 0
        print('error {}'.format(e))

    for i_episode in range(200):
        for i_batch, sample_batch in enumerate(data_loader):
            (X_train, Y_train) = sample_batch
            X_train = X_train.transpose(1, 3)
            output = model(X_train)
            loss = criterion(output, Variable(Y_train.long()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('episode {} batch {} loss is : {}'.format(i_episode, i_batch, loss))

        writer.add_scalar('loss', loss, global_step)
        global_step += 1

        if 0 == i_episode % 20:
            print("before save global_step is {}".format(global_step))
            torch.save({'step': global_step, 'model': model}, MODE_FILE_PATH)
    torch.save({'step': global_step, 'model': model}, MODE_FILE_PATH)
