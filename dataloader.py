from torch.utils.data import Dataset
import os
from tokenizer import SPTokenizer
import config
import json

class PretrainDataset(Dataset):
    def __init__(self):
        self.file_path = config.PRETRAIN_DATA_PATH
        self.files=[]
        self.indexes=[0]
        for file in os.listdir(self.file_path):
            self.files.append(self.file_path+"/"+file)
            with open(self.file_path+"/"+file) as fp:
                num=len(json.load(fp))
            self.indexes.append(self.indexes[-1]+num-1)
        self.tokenizer=SPTokenizer(config.VOCAB_FILE)

    def __len__(self):
        return self.indexes[-1]

    def __getitem__(self, idx):
        file_idx=self.find_interval(idx)
        print(file_idx)
        cur_file=self.files[file_idx]
        with open(cur_file) as fp:
            text=json.load(fp)[idx-self.indexes[file_idx]]
        return self.tokenizer.encode(text["title"]+"\n\n"+text["content"])

    def find_interval(self, target):
        arr=self.indexes
        n = len(arr)
        # 处理目标数小于列表中最小数的情况
        if target <= arr[0]:
            return 0
        # 处理目标数大于列表中最大数的情况
        elif target >= arr[-1]:
            return arr[-1]
        else:
            # 使用二分搜索定位target可能的区间
            left, right = 0, n - 1
            while left <= right:
                mid = left + (right - left) // 2
                # 如果找到了目标数，直接返回其周围的两个数
                if arr[mid] == target:
                    # 处理target为列表中最后一个元素的情况
                    if mid == n - 1:
                        return arr[mid - 1]
                    else:
                        return arr[mid - 1]
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            # 如果target不在列表中，left和right会相邻，并且right < left
            return right

    def test(self):
        print(self.__len__())
        print(self.indexes)
        print(self.__getitem__(10000))
        print(self.__getitem__(9999))
        print(self.__getitem__(10001))

dd=PretrainDataset()
dd.test()