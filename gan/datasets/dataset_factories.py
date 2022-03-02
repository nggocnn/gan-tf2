from gan.datasets import cifar10
from gan.datasets import fashion_mnist
from gan.datasets import mnist
from gan.datasets.dataset_type import DatasetType
from gan.datasets import summer_to_winter


def get_dataset(input_params, dataset_type: DatasetType):
    if dataset_type == DatasetType.VANILLA_MNIST.name:
        return mnist.MnistDataset(input_params)

    elif dataset_type == DatasetType.VANILLA_FASHION_MNIST.name:
        return fashion_mnist.FashionMnistDataset(input_params)

    elif dataset_type == DatasetType.VANILLA_CIFAR10.name:
        return cifar10.Cifar10Dataset(input_params)

    elif dataset_type == DatasetType.CONDITIONAL_MNIST.name:
        return mnist.MnistDataset(input_params, with_labels=True)

    elif dataset_type == DatasetType.CONDITIONAL_FASHION_MNIST.name:
        return fashion_mnist.FashionMnistDataset(input_params, with_labels=True)

    elif dataset_type == DatasetType.CONDITIONAL_CIFAR10.name:
        return cifar10.Cifar10Dataset(input_params, with_labels=True)

    elif dataset_type == DatasetType.CYCLE_SUMMER2WINTER.name:
        return summer_to_winter.SummerToWinterDataset(input_params)

    else:
        raise NotImplementedError
