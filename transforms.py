import random
from torchvision.transforms import functional as F
from PIL import Image

class RandomResize(object):
    """Resize the object randomly between the min and max size"""

    def __init__(self, min_size:int, max_size:int, is_mask:bool = False):
        """min size is the minimum desired size of the image
        max_size is the maximum desired size of the image
        is_mask isto check if the image is the segmentation mask"""

        self.min_size = min_size
        self.max_size = max_size
        self.is_mask = is_mask

    def __call__(self, img:Image) -> Image:
        size = random.randint(self.min_size, self.max_size)
        if self.is_mask:
            return F.resize(img, size=size, interpolation=Image.NEAREST)
        else:
            return F.resize(img, size = size)
        
