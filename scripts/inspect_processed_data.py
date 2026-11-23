import numpy as np
import torch
import sys

# with np.load('data/processed/supervised.npz', allow_pickle=False) as data:
#     X_goes = data['X_goes']
#     X_numerical = data['X_numerical']
#     Y_rain_mm = data['Y_rain_mm']
#     goes_valid = data['X_goes_mask']
#     numerical_valid = data['X_num_mask']
#     rain_valid = data['Y_rain_mask']
#     times = data['issuance_times_utc']

# print(type(X_numerical), np.shape(X_numerical))
# print(type(numerical_valid), np.shape(numerical_valid))
# print(type(times), np.shape(times))
# print(X_numerical[0][0][0])
# print(numerical_valid[0][0][0])

def get_numerical_data_ndarray(lbound: int, rbound: int):
    with np.load('data/processed/supervised.npz', allow_pickle=False) as data:
        X_numerical = data['X_numerical']
        
    # X_numerical = np.transpose(X_numerical, (0,2,1,3,4)) # (nSamples)->(weather_attrs)->(future_forecast)->(xcoord)->(ycoord)
    n, w, f, x, y = np.shape(X_numerical)
    
    #fold time axis into channel; we do this because there are only 2 forecasts
    #in the ground truth rain data
    X_numerical = np.reshape(X_numerical, shape=(n, w * f, x, y))
   
    return X_numerical[lbound:rbound]

def get_rain_data_ndarray(lbound: int, rbound: int):
    with np.load('data/processed/supervised.npz', allow_pickle=False) as data:
        Y_rain_mm = data['Y_rain_mm']   #comes in a (nsamples)->(future_forecast)->(xcoord)->(ycoord)

    

    # Y_rain_mm = Y_rain_mm[np.newaxis, ...]  #append a 1-depth axis to match input shape
    # Y_rain_mm = np.transpose(Y_rain_mm, tuple([1, 0, 2, 3, 4])) #(nsamples)->(1depth)->(time)->(xcoord)->(ycoord) 
    # modified_shape = np.shape(Y_rain_mm)
    # modified_shape = list(modified_shape)
    # modified_shape[2] = 3
    # modified_shape = tuple(modified_shape)
    # copy = np.zeros(shape=modified_shape)
    # np.copyto(copy, Y_rain_mm)
    # Y_rain_mm = copy

    # print(Y_rain_mm[0][1])

    return Y_rain_mm[lbound:rbound]

def get_numerical_data_and_validity_ndarray(lbound: int, rbound: int):
    with np.load('data/processed/supervised.npz', allow_pickle=False) as data:
        numerical_valid = data['X_num_mask']

    n, w, f, x, y = np.shape(numerical_valid)
    
    #fold time axis into channel; we do this because there are only 2 forecasts
    #in the ground truth rain data
    numerical_valid = np.reshape(numerical_valid, shape=(n, w * f, x, y))
   
    X_numerical = get_numerical_data_ndarray(lbound, rbound)

    #we establish validity as its own axis
    numerical_validity_wrapper = np.array([X_numerical, numerical_valid[lbound:rbound]])
    numerical_validity_wrapper = np.transpose(numerical_validity_wrapper, (1, 2, 0, 3, 4))

    print("Returning nuemrical + validity", np.shape(numerical_validity_wrapper))
    return numerical_validity_wrapper

def get_rain_data_and_validity_ndarray(lbound: int, rbound:int):
    Y_rain_mm =  get_rain_data_ndarray(lbound, rbound)

    with np.load('data/processed/supervised.npz', allow_pickle=False) as data:
        rain_valid = data['Y_rain_mask']

    #validity is its own axis
    rain_validity_wrapper = np.array([Y_rain_mm, rain_valid[lbound:rbound]])
    rain_validity_wrapper = np.transpose(rain_validity_wrapper, (1, 2, 0, 3, 4))

    print("Returning rain + validity", np.shape(rain_validity_wrapper))
    return rain_validity_wrapper