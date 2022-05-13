import numpy as np
from torchvision import datasets
from default_param import CITYSCAPES_PATH


class CityscapesUtils:
    """This utility class provides the mapping of trainin labels and corresponding colors for visualization

    """
    def __init__(self):
        cityscapes_data = datasets.Cityscapes(CITYSCAPES_PATH, split = 'train', mode = 'fine', target_type = 'semantic')
        self.classes = cityscapes_data.classes()
        self.num_classes = self._num_classes()
        self.train_id2color = self._train_id2color()
        self.id2train_id = self.id2train_id()


    def _num_classes(self) -> int:
        """returns the number of classes in cityscapes that are used in validation"""
        train_labels = [label.id for label in self.classes if not label.ignore_in_eval]
        return len(train_labels)

    def _id2train_id(self) -> np.array:
        """returns a list where indexes of the list is mapped to its training_id 
        0 index are correspondent/mapped to unlabelled classes"""
        train_ids = np.array([label.train_id for label in self.classes])
        train_ids[(train_ids == -1) | (train_ids == 255)] = 19 #19 is Ignore_Index (defaults.CITYSCAPES_IGNORE_INDEX
        return train_ids

    def _train_id2color(self) -> np.array:
        """This returns the mapping of 20 classes (19 training classes and 1 ignore index class) to standard color used in cityscapes"""

        return np.array([label.color for label in self.classes if label.ignore_in_eval] + [(0, 0, 0)])


    def label2color(self, mask: np.array) -> np.array:
        return self.train_id2color[mask]
