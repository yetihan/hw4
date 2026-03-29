import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        if train:
            files = [x for x in os.listdir(base_folder) if 'data' in x]
        else:
            files = [x for x in os.listdir(base_folder) if 'test' in x]
        assert len(files)>0, f"no data in {base_folder}"
        for file_name in files:
            with open(os.path.join(base_folder, file_name), 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                data, label = dict[b'data']/255, dict[b'labels']
                s = data.shape
                data = data.reshape(s[0],3,32,32)
                if 'X' not in locals():
                    X, y = data, np.array(label)
                else:    
                    X = np.concatenate([X, data], axis=0)
                    y = np.concatenate([y, label], axis=0)
        self.X = X 
        self.y = y
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        return self.X[index], self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.y.shape[0]
        ### END YOUR SOLUTION
