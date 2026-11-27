import numpy as np
import torch
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from torch.utils.data.dataset import Dataset
import sys

# with np.load('data/processed/supervised.npz', allow_pickle=False) as data:
#     X_goes = data['X_goes']
#     X_numerical = data['X_numerical']
#     Y_rain_mm = data['Y_rain_mm']
#     goes_valid = data['X_goes_mask']
#     numerical_valid = data['X_num_mask']
#     rain_valid = data['Y_rain_mask']
#     times = data['issuance_times_utc']

def get_numerical_data_ndarray(lbound: int, rbound: int):
    with np.load('data/processed/supervised.npz', allow_pickle=False) as data:
        X_numerical = data['X_numerical']

    n, f, w, x, y = np.shape(X_numerical)   #(nSamples)->(historical_interval)->(weather_attrs)->(xcoord)->(ycoord)
    
    #fold time axis into channel; we do this because there 3 historical intervals backwards in input
    #but only 2 intervals forward for output in the ground truth rain data
    X_numerical = np.reshape(X_numerical, shape=(n, f * w, x, y))
   
    return X_numerical[lbound:rbound]

def get_rain_data_ndarray(lbound: int, rbound: int):
    with np.load('data/processed/supervised.npz', allow_pickle=False) as data:
        Y_rain_mm = data['Y_rain_mm']   #comes in a (nsamples)->(future_forecast)->(xcoord)->(ycoord)

    #we can return it as is
    return Y_rain_mm[lbound:rbound]

def get_numerical_data_and_validity_ndarray(lbound: int, rbound: int):
    with np.load('data/processed/supervised.npz', allow_pickle=False) as data:
        numerical_valid = data['X_num_mask']

    n, f, w, x, y = np.shape(numerical_valid)

    #same folding for the numerical data
    numerical_valid = np.reshape(numerical_valid, shape=(n, f * w, x, y))
   
    X_numerical = get_numerical_data_ndarray(lbound, rbound)

    #we establish "validity" as its own axis - [0] for attribute values, [1] for attribute validity
    numerical_validity_wrapper = np.array([X_numerical, numerical_valid[lbound:rbound]])
    numerical_validity_wrapper = np.transpose(numerical_validity_wrapper, (1, 2, 0, 3, 4))

    #fold the validity axis back into the channel
    n, w, v, x, y = np.shape(numerical_validity_wrapper)
    numerical_validity_wrapper = np.reshape(numerical_validity_wrapper, (n, w * v, x, y))

    print("Returning numerical + validity", np.shape(numerical_validity_wrapper))
    return numerical_validity_wrapper

def get_rain_data_and_validity_ndarray(lbound: int, rbound:int):
    Y_rain_mm =  get_rain_data_ndarray(lbound, rbound)

    with np.load('data/processed/supervised.npz', allow_pickle=False) as data:
        rain_valid = data['Y_rain_mask']

    #validity is its own axis, see above reason
    rain_validity_wrapper = np.array([Y_rain_mm, rain_valid[lbound:rbound]])
    rain_validity_wrapper = np.transpose(rain_validity_wrapper, (1, 2, 0, 3, 4))

    #fold the validity axis back into the channel
    n, f, v, x, y = np.shape(rain_validity_wrapper)
    rain_validity_wrapper = np.reshape(rain_validity_wrapper, (n, f * v, x, y))

    print("Returning rain + validity", np.shape(rain_validity_wrapper))
    return rain_validity_wrapper

def get_goes_data_ndarray(lbound: int, rbound:int):
    with np.load('data/processed/supervised.npz', allow_pickle=False) as data:
        X_goes = data['X_goes']

    n, f, w, x, y = np.shape(X_goes) #(n_samples)->(historical_interval)
    
    #fold time axis into channel; in effect nothing should really change
    X_goes = np.reshape(X_goes, shape=(n, w * f, x, y))
   
    return X_goes[lbound:rbound]

def get_goes_data_and_validity_ndarray(lbound: int, rbound: int):
    X_goes = get_goes_data_ndarray(lbound, rbound)

    with np.load('data/processed/supervised.npz', allow_pickle=False) as data:
        goes_valid = data['X_goes_mask']

    n, f, w, x, y = np.shape(goes_valid) #(n_samples)->(historical_interval)
    
    #fold time axis into channel; in effect nothing should really change
    goes_valid = np.reshape(goes_valid, shape=(n, f * w, x, y))

    #validity is its own axis once more
    goes_validity_wrapper = np.array([X_goes, goes_valid[lbound:rbound]])
    goes_validity_wrapper = np.transpose(goes_validity_wrapper, (1, 2, 0, 3, 4))

    #but then fold it back in
    #fold the validity axis back into the channel
    n, w, v, x, y = np.shape(goes_validity_wrapper)
    goes_validity_wrapper = np.reshape(goes_validity_wrapper, (n, w * v, x, y))


    print("Returning GOES + validity", np.shape(goes_validity_wrapper))
    return goes_validity_wrapper

def provide_normalizer(dataset : Dataset, key : str, channels : int):
    #there is a quirk with the UnitGaussianNormalizer library
    #that prevents exclusion of certain channels in normalization
    #this solution is needed as a result.

    #provides a unit_gaussian normalizer
    sum_x = torch.zeros((1, int(channels/2), 1, 1), dtype=torch.float64)
    sum_x2 = torch.zeros((1, int(channels/2), 1, 1), dtype=torch.float64)
    valid_count = torch.zeros((1, int(channels/2), 1, 1), dtype=torch.float64)

    for sample in dataset:
        x = sample[key].to(torch.float64)
        if x.ndim == 3:
            x = x.unsqueeze(0)  # (1, 30, 32, 32)

        data = x[:, 0::2]   # even channels: 15 numerical variables
        valid = x[:, 1::2]  # odd channels: 15 binary validity masks

        sum_x += (data * valid).sum(dim=(0, 2, 3), keepdim=True)
        sum_x2 += ((data ** 2) * valid).sum(dim=(0, 2, 3), keepdim=True)
        valid_count += valid.sum(dim=(0, 2, 3), keepdim=True)

    safe_count = valid_count.clamp(min=1.0)
    data_mean = (sum_x / safe_count).to(torch.float32)
    data_var = ((sum_x2 / safe_count) - (sum_x / safe_count) ** 2).clamp(min=1e-12)
    data_std = torch.sqrt(data_var).to(torch.float32)

    # default mean=0, std=1 keeps odd validity channels (1::2) untouched
    mean = torch.zeros((1, channels, 1, 1), dtype=torch.float32)
    std = torch.ones((1, channels, 1, 1), dtype=torch.float32)
    mean[:, 0::2] = data_mean
    std[:, 0::2] = data_std

    return UnitGaussianNormalizer(mean=mean, std=std)