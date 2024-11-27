import pathlib
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torchvision.io import read_image
from sklearn.model_selection import train_test_split


class ImageDataset(Dataset):
    def __init__(self, file_list, labels):
        self.file_list = file_list
        self.labels = labels
    def __getitem__(self, index):
        file = self.file_list[index]
        image = read_image(file)
        image = image / 255 # convert to (minmax) float for compatibility
        label = self.labels[index]
        return image, label
    def __len__(self):
        return len(self.labels)
    def dim(self):
        return self[0][0].shape

class ConvNet(nn.Module):
    def __init__(self, input_dim, output_channels, batch_size = 16):
        super().__init__()

        self.feature_extractor = nn.Sequential(

            nn.Conv2d(
                in_channels=3, out_channels=32,
                kernel_size=5, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=5, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=64, out_channels=128,
                kernel_size=5, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        channels, h, w = input_dim
        x = torch.ones((batch_size, channels, h, w))
        out_dim = self.feature_extractor(x).shape
        #print(out_dim)
        #num_features_before_fcnn = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(batch_size, *input_dim)).shape))
        self.classifier = nn.Sequential(
            nn.Linear(out_dim[1], 1024),
            nn.ReLU(),
            nn.Linear(1024, 400),
            nn.ReLU(),
            nn.Linear(400, output_channels)
        )
        #print(self.classifier(self.feature_extractor(x)).shape)


    def forward(self, x):
        out = self.feature_extractor(x)
        out = self.classifier(out)
        return out


def img_label_from_folder(path):
    """
    Get list of files with labels from the specific folder
    structure in our project. For now extracting labels
    is hardcoded, and requires named folders 4 levels deep.
    Feel free to come up with a better implementation.
    """
    imgdir_path = pathlib.Path(path)
    file_list = sorted([str(path) for path in imgdir_path.rglob('*.jpg')])

    category = [path.split("/")[3] for path in file_list]

    label_dict = {cat:i for i, cat in enumerate(set(category))}
    labels = [label_dict[cat] for cat in category]

    return file_list, labels, label_dict

def split_imagedata(file_list, labels, test_size=0.2, valid_size=0.2):
    """
    Takes a list of image files and labels, splits in train, test and
    validation set. Returns an ImageDataset instance for each set.
    Test and valid size are given as proportions of the full data set.
    """
    valid_size_of_train = valid_size / (1 - test_size)

    img_train_, img_test, label_train_, label_test = train_test_split(file_list, labels, test_size=test_size)
    img_train, img_valid, label_train, label_valid = train_test_split(img_train_, label_train_, test_size=valid_size_of_train)

    train_set = ImageDataset(img_train, label_train)
    valid_set = ImageDataset(img_valid, label_valid)
    test_set = ImageDataset(img_test, label_test)

    return train_set, valid_set, test_set

def train_cnn(model, num_epochs, train_dl, valid_dl, optimizer, device, loss_fn = nn.CrossEntropyLoss()):
    """
    Train the CNN, from Raschka et al
    """
    model.to(device)
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item()*y_batch.size(0)
            is_correct = (
                torch.argmax(pred, dim=1) == y_batch
                ).float()
            accuracy_hist_train[epoch] += is_correct.sum()
        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item()*y_batch.size(0)
                is_correct = (
                    torch.argmax(pred, dim=1) == y_batch
                ).float()
                accuracy_hist_valid[epoch] += is_correct.sum()

        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

        print(f'Epoch {epoch+1} accuracy: '
            f'{accuracy_hist_train[epoch]:.4f} val_accuracy: '
            f'{accuracy_hist_valid[epoch]:.4f}')
        
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid