import random
import numpy as np
from paddle import fluid

class DataLoader:
    place = fluid.CPUPlace()
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.loader = fluid.io.DataLoader.from_generator(capacity=32, return_list=True, use_double_buffer=True, use_multiprocess=False)
    def __iter__(self):
        self.loader.set_sample_generator(self.sample_generator_creator, batch_size=self.batch_size, drop_last=True, places=DataLoader.place)
        for item in self.loader:
            yield item
    def sample_generator_creator(self):
        len_dataset = len(self.dataset)
        idx = list(range(len_dataset))
        if self.shuffle:
            random.shuffle(idx)
        for i in idx:
            yield self.dataset[i]
    def concat_datas(self, datas):
        '''
        input: [(img1, label1), (img2, label2)]
        output: [stack(img1, img2), stack(label1, label2)]
        '''
        out = []
        for t in range(len(datas[0])):
            li = [data[t] for data in datas]
            out.append(np.stack(li))
        return out
