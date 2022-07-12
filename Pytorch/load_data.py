from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from utils import create_bb_array, create_mask
from torch.utils.data import random_split
import torchvision.transforms.functional as TF
import xml.etree.ElementTree as ET

class SteelDefect(Dataset):
    def __init__(self, image_path_list, mask_path_list, augmentation=None):
        super().__init__()
        self.image_path_list = image_path_list
        self.mask_path_list = mask_path_list
        self.augmentation = augmentation
        
        
    def __len__(self):
        return len(self.image_path_list)
    
    def __getitem__(self, idx):

        image = Image.open(self.image_path_list[idx])
        tree = ET.parse(self.mask_path_list[idx])
        root = tree.getroot()
        i = root.findall('object')
        
        width = int(root.find('size')[0].text)
        height = int(root.find('size')[1].text)
        bb = create_bb_array(i[0],width, height)

        mask = create_mask(bb, image)
        mask = Image.fromarray(mask)
        
        
        if self.augmentation:
            sample = self.augmentation(image=np.asarray(image), mask=np.asarray(mask))
            image, mask = sample['image'], sample['mask']

        
            image, mask = TF.resize(Image.fromarray(image), (320,320)), TF.resize(Image.fromarray(mask), (320,320))
            image, mask = TF.to_tensor(image), TF.to_tensor(mask)
        else:
            image, mask = TF.resize(image, (320,320)), TF.resize(mask, (320,320))
            image, mask = TF.to_tensor(image), TF.to_tensor(mask)
        
        mask = (mask >= 0.5)*(1.0)
        
        return image, mask