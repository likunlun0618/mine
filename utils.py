import os
import json
import numpy as np

# use('Agg')是为了在无桌面环境时也可以生成图片
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class Logger(object):

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.data_dict = {}
        os.makedirs(log_dir, exist_ok=True)

    def record(self, *args, log10=False):
        assert(len(args)%2 == 0)
        for i in range(len(args)//2):
            key, value = args[i*2], args[i*2+1]
            assert(type(key) == str)
            assert(type(value) == float or type(value) == int)

            if log10:
                # 将value求log10之后再保存，用于value变化特别大的情况
                value = np.log10(value)

            if key in self.data_dict:
                self.data_dict[key].append(np.log10(value))
            else:
                self.data_dict[key] = [np.log10(value)]

    def save_fig(self, *keys, avg=1):
        for key in keys:
            if key not in self.data_dict:
                continue
            n = len(self.data_dict[key]) // avg
            data = []
            for i in range(n):
                data.append(sum(self.data_dict[key][i*avg: (i+1)*avg]) / avg)
            plt.plot(range(n), data, label=key)
            plt.legend()
            plt.savefig(self.log_dir+key+'.png')
            plt.clf()

    def save_json(self, *keys):
        for key in keys:
            if key not in self.data_dict:
                continue
            with open(self.log_dir+key+'.json', 'w') as f:
                json.dump(self.data_dict[key], f)

    def clear(self, *keys):
        if len(keys) == 0:
            self.data_dict = {}
        else:
            for key in keys:
                if key in self.data_dict:
                    self.data_dict.pop(key)
