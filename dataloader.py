from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import scipy.io as sio
import torch
import os
class DataSet_NoisyMNIST(object):

    def __init__(self, images1, images2, labels, fake_data=False, one_hot=False,
                 dtype=np.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        if dtype not in (np.uint8, np.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images1.shape[0] == labels.shape[0], (
                    'images1.shape: %s labels.shape: %s' % (images1.shape,
                                                            labels.shape))
            assert images2.shape[0] == labels.shape[0], (
                    'images2.shape: %s labels.shape: %s' % (images2.shape,
                                                            labels.shape))
            self._num_examples = images1.shape[0]

            if dtype == np.float32 and images1.dtype != np.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                print("type conversion view 1")
                images1 = images1.astype(np.float32)

            if dtype == np.float32 and images2.dtype != np.float32:
                print("type conversion view 2")
                images2 = images2.astype(np.float32)

        self._images1 = images1
        self._images2 = images2
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images1(self):
        return self._images1

    @property
    def images2(self):
        return self._images2

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in range(batch_size)], [fake_image for _ in range(batch_size)], [fake_label for _
                                                                                                      in range(
                    batch_size)]

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images1 = self._images1[perm]
            self._images2 = self._images2[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self._images1[start:end], self._images2[start:end], self._labels[start:end]
class NoisyMNIST(Dataset):
    def __init__(self, path):
        """
        初始化 NoisyMNIST 数据集。

        :param path: 数据文件的路径
        """
        try:
            # 加载带噪声的 MNIST 数据集
            data = sio.loadmat(path + 'NoisyMNIST.mat')  # 从 .mat 文件中加载数据
        except FileNotFoundError:
            raise FileNotFoundError("数据文件未找到，请检查路径。")
        except Exception as e:
            raise Exception(f"加载数据时发生错误: {e}")

        # 创建训练集、调优集和测试集的实例
        train = DataSet_NoisyMNIST(data['X1'], data['X2'], data['trainLabel'])  # 创建训练数据集
        tune = DataSet_NoisyMNIST(data['XV1'], data['XV2'], data['tuneLabel'])  # 创建调优数据集
        test = DataSet_NoisyMNIST(data['XTe1'], data['XTe2'], data['testLabel'])  # 创建测试数据集

        # 初始化图像和标签
        self.X1 = np.concatenate([train.images1, tune.images1, test.images1], axis=0)  # 合并训练、调优和测试集的第一个视图图像
        self.X2 = np.concatenate([train.images2, tune.images2, test.images2], axis=0)  # 合并训练、调优和测试集的第二个视图图像
        self.Y = np.concatenate([np.squeeze(train.labels[:, 0]), np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])])  # 合并训练、调优和测试集的标签

        # 获取样本数量
        self.num_samples = self.X1.shape[0]  # 获取样本数量

    def __len__(self):
        """返回数据集的样本数量。"""
        return self.num_samples  # 返回数据集的样本数量

    def __getitem__(self, idx):
        """根据索引获取样本。

        :param idx: 样本索引
        :return: (X1, X2): 图像视图，标签，索引
        """
        # 检查索引是否在有效范围内
        if idx >= self.num_samples or idx < 0:
            raise IndexError("索引超出范围")

        # 根据索引获取样本
        X1 = self.X1[idx].reshape(784)  # 获取第一个视图的图像并调整形状为 (784,)
        X2 = self.X2[idx].reshape(784)  # 获取第二个视图的图像并调整形状为 (784,)

        # 返回两个视图的图像、对应的标签和索引
        return [torch.from_numpy(X1), torch.from_numpy(X2)], self.Y[idx], torch.tensor(idx).long()


class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
class Hdigit(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Hdigit.mat')
        scaler = MinMaxScaler()
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(10000,)
        self.V1 = scaler.fit_transform(data['data'][0][0].T.astype(np.float32))
        self.V2 = scaler.fit_transform(data['data'][0][1].T.astype(np.float32))

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()




class Caltech101_20(Dataset):
    def __init__(self, path):
        # 加载数据
        mat = scipy.io.loadmat(os.path.join(path, 'Caltech101-20.mat'))
        X = mat['X'][0]  # 获取数据集中的所有视图
        self.x = []  # 用于存储规范化后的每个视图数据

        # 选择视图 3 和 4，并进行规范化
        for view in [3, 4]:
            x = X[view]
            x = self.normalize(x).astype(np.float32)  # 规范化
            self.x.append(x)  # 将每个视图的数据添加到列表中

        self.y = np.squeeze(mat['Y']).astype(np.int32)  # 获取标签

    def normalize(self, x):
        # 规范化函数
        return (x - np.mean(x)) / np.std(x)

    def __len__(self):
        return len(self.y)  # 返回标签的长度

    def __getitem__(self, idx):
        # 获取样本和标签
        x1 = torch.from_numpy(self.x[0][idx])  # 获取视图 1 的数据
        x2 = torch.from_numpy(self.x[1][idx])  # 获取视图 2 的数据
        y = torch.from_numpy(np.array(self.y[idx])).long()  # 获取标签
        return [x1, x2], y, torch.tensor(idx).long()  # 返回数据、标签和索引

def load_data(dataset):
    if dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "NoisyMNIST":
        dataset = NoisyMNIST('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 70000#70000
    elif dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Fashion":
        dataset = Fashion('./data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "Caltech-2V":
        dataset = Caltech('data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech101_20":
        dataset = Caltech101_20('./data/')
        dims = []
        view = 2
        data_size = len(dataset)
        class_num = 101
    elif dataset == "Hdigit":
        dataset = Hdigit('data/')
        dims = [784, 256]
        view = 2
        data_size = 10000
        class_num = 10
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num