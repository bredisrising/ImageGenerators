import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

id_to_class = {
    0: "Audi",
    1: "Hyundai",
    2: "Rolls Royce",
    3: "Swift",
    4: "Toyota"
}

class AllVae(Dataset):
    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def __len__(self):
        return 16
    
    def __getitem__(self, index):
        x = Image.open(f'./data/cars/all_processed_64px/{index+1}.jpg')
        x = self.transform(x)
        return x

class AudiAutoregressionGrayscale24px(Dataset):
    def __init__(self):
        # grayscale transform but keep 256 range
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])


    def __len__(self):
        return 72 #1628

    def without_mask(self, index):
        x = Image.open(f"./data/cars/Audi_processed_24px/{index+1}.jpg")
        x = self.transform(x)
        return x.view((1, 24, 24))

        

    def get_specific_mask(self, index, mask):
        # load image
        x = Image.open(f'./data/cars/Audi_processed_24px/{index+1}.jpg')
        x = self.transform(x)
        x = torch.reshape(x, (1, 24*24))
        # x = torch.flatten(x)
        
        to_predict = [mask]

        target1 = x[0][to_predict[0]].clone().unsqueeze(0)

        pixel_value = torch.tensor([target1 * 255.0]).to(torch.long)
        pixel_value = pixel_value.unsqueeze(0)
        #pixel_value = torch.nn.functional.one_hot(pixel_value, num_classes=256).squeeze(0).float()


        x[0][to_predict[0]:] = 0.0
        #x[0][to_predict[0]] = -1.0
        
        x = torch.flatten(x)

        # position_to_predict = to_predict[0] / 256.0

        # # print(x, x.shape)
        # # print(position_to_predict, position_to_predict.shape)

        # x = torch.cat((position_to_predict.unsqueeze(0), x))

        return x, pixel_value

    def __getitem__(self, index):
        # load image
        x = Image.open(f'./data/cars/Audi_processed_24px/{index+1}.jpg')
        x = self.transform(x)
        x = torch.reshape(x, (1, 24*24))
        # x = torch.flatten(x)
        
        to_predict = torch.randint(0, 24*24-1, (1,))

        target1 = x[0][to_predict[0]].clone().unsqueeze(0)

        pixel_value = torch.tensor([target1 * 255.0]).to(torch.long)
        pixel_value = pixel_value.unsqueeze(0)
        #pixel_value = torch.nn.functional.one_hot(pixel_value, num_classes=256).squeeze(0).float()


        x[0][to_predict[0]:] = 0.0
        #x[0][to_predict[0]] = -1.0

        # position_to_predict = to_predict[0] / 256.0

        # # print(x, x.shape)
        # # print(position_to_predict, position_to_predict.shape)

        # x = torch.cat((position_to_predict.unsqueeze(0), x))

        return x.view((1, 24, 24)), pixel_value

class AllAutoregressionColor24px(Dataset):
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