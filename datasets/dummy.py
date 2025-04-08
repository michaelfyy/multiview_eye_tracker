import torch
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __init__(self, num_samples: int = 100, image_size: tuple = (3, 224, 224), transform=None):
        """
        Creates a dummy dataset with random pixel values and dummy regression labels.
        
        Args:
            num_samples (int): Number of dummy samples.
            image_size (tuple): The size of each image, e.g. (3, 224, 224).
            transform (callable, optional): Optional transform to be applied on the image.
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random image with pixel values in [0, 1]
        image = torch.rand(self.image_size)
        
        # Create dummy regression labels:
        # For pupil localization: a 3D point (x, y, z)
        # For gaze regression: a 3D unit vector (could be any dummy values)
        pupil_label = torch.rand(3)
        random_vector = torch.rand(3)
        gaze_label = random_vector / torch.norm(random_vector)
        
        sample = {'image': image, 'pupil': pupil_label, 'gaze': gaze_label}
        
        if self.transform:
            # Note: if your transform expects a PIL Image, you may need to convert the tensor
            sample['image'] = self.transform(image)
            
        return sample