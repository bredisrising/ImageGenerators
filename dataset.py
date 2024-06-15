import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class AudiAutoregression24px(Dataset):
    def __init__(self):
        # grayscale transform
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return 5190

    def __getitem__(self, index):
        # load image
        x = Image.open(f'./data/cars/all_processed_24px/{index+1}.jpg')
        x = self.transform(x)
        x = torch.reshape(x, (3, 24*24))
        # x = torch.flatten(x)
        
        to_predict = torch.randint(0, 24*24-1, (1,))

        target1 = x[0][to_predict[0]].clone().unsqueeze(0)
        x[0][to_predict[0]:] = 0.0
        x[0][to_predict[0]] = -1.0

        target2 = x[1][to_predict[0]].clone().unsqueeze(0)
        x[1][to_predict[0]:] = 0.0
        x[1][to_predict[0]] = -1.0
        
        target3 = x[2][to_predict[0]].clone().unsqueeze(0)
        x[2][to_predict[0]:] = 0.0
        x[2][to_predict[0]] = -1.0
        
        x = torch.flatten(x)

        # position_to_predict = to_predict[0] / 256.0

        # # print(x, x.shape)
        # # print(position_to_predict, position_to_predict.shape)

        # x = torch.cat((position_to_predict.unsqueeze(0), x))

        return x, torch.cat((target1, target2, target3))



class AudiAutoregression16px(Dataset):
    def __init__(self):
        # grayscale transform
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return 1628

    def __getitem__(self, index):
        # load image
        x = Image.open(f'./data/cars/Audi_processed_16px/{index+1}.jpg')
        x = self.transform(x)
        x = torch.reshape(x, (3, 256))
        # x = torch.flatten(x)
        
        to_predict = torch.randint(0, 255, (1,))

        target1 = x[0][to_predict[0]].clone().unsqueeze(0)
        x[0][to_predict[0]:] = 0.0
        x[0][to_predict[0]] = -1.0

        target2 = x[1][to_predict[0]].clone().unsqueeze(0)
        x[1][to_predict[0]:] = 0.0
        x[1][to_predict[0]] = -1.0
        
        target3 = x[2][to_predict[0]].clone().unsqueeze(0)
        x[2][to_predict[0]:] = 0.0
        x[2][to_predict[0]] = -1.0
        
        x = torch.flatten(x)

        # position_to_predict = to_predict[0] / 256.0

        # # print(x, x.shape)
        # # print(position_to_predict, position_to_predict.shape)

        # x = torch.cat((position_to_predict.unsqueeze(0), x))

        return x, torch.cat((target1, target2, target3))