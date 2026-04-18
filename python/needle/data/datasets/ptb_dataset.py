import os

import numpy as np
from needle import backend_ndarray as nd
from needle import Tensor

class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        if word not in self.word2idx:
            self.word2idx[word]=self.__len__()
            self.idx2word.append(word)
        return self.word2idx[word]
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return self.idx2word.__len__()
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        id_lst = []        
        with open(path) as f:
            for i, line in enumerate(f):
                if max_lines is not None and i>=max_lines:
                    break
                for word in line.split():
                    id_lst.append(self.dictionary.add_word(word))
                id_lst.append(self.dictionary.add_word('<eos>'))
        return id_lst
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    N = len(data)
    nbatch = N//batch_size
    # res = [[] for _ in  range(nbatch)]
    # 三种等价写法，核心都是把 1D token 序列切成 batch_size 段，每段作为一列
    #
    # 写法 1: 双循环显式填充 —— 最直观，但 Python 循环慢
    #   res[i][j] = data[j*nbatch + i]  → 第 j 列是 data[j*nbatch : (j+1)*nbatch]
    # res = [[data[j*nbatch+i] for j in range(batch_size)] for i in range(nbatch)]
    # res = np.array(res, dtype=dtype)
    #
    # 写法 2: Fortran-order reshape —— 按列优先填入 (nbatch, batch_size)
    #   stride_j = nbatch, stride_i = 1，正好对应 data[j*nbatch+i]
    # res = np.array(data[:nbatch*batch_size], dtype=dtype).reshape(
    #     (nbatch, batch_size), order='F')
    #
    # 写法 3: C-order reshape + 转置 —— 最常见写法（PyTorch 官方示例即此写法）
    #   先 reshape(batch_size, nbatch)：每行一段连续 token
    #   再 .T：转成 (nbatch, batch_size)，每列一段连续 token
    res = np.array(data[:nbatch*batch_size], dtype=dtype).reshape(batch_size, nbatch).T
    return np.array(res, dtype=dtype)
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    data = batches[i:i+bptt]
    target = batches[i+1:i+bptt+1].reshape(np.prod(data.shape))
    return Tensor(data, device=device, dtype=dtype), Tensor(target, device=device, dtype=dtype),
    
    ### END YOUR SOLUTION