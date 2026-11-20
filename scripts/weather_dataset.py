from torch.utils.data.dataset import Dataset, TensorDataset
from torch import Tensor
from torch import FloatTensor

class NumericalWeatherDataset(Dataset):

    #list containing tuple of corresponding numerical->precipitation pairs
    sample_list: list[tuple[Tensor, Tensor]]

    #fill the sample list - "unfold" the tensors by one dimension to use as a list bases
    def __init__(self, x : Tensor, y: Tensor):
        self.sample_list = list(zip(x, y))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index : int):
        numerical_tensor, rain_tensor = self.sample_list[index]
        return {
            "x" : numerical_tensor,
            "y" : rain_tensor
        }