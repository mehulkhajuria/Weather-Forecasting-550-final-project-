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
    X_numerical = np.transpose(X_numerical, (0,2,1,3,4)) # (nSamples)->(weather_attrs)->(future_forecast)->(xcoord)->(ycoord)

    #such that the three source axes are time, grid_x, grid_y
    
    return X_numerical[lbound:rbound]

def get_rain_data_ndarray(lbound: int, rbound: int):
    with np.load('data/processed/supervised.npz', allow_pickle=False) as data:
        Y_rain_mm = data['Y_rain_mm']   #comes in a (nsamples)->(future_forecast)->(xcoord)->(ycoord)

    

    # Y_rain_mm = Y_rain_mm[np.newaxis, ...]  #append an axis to match input shape
    # Y_rain_mm = np.transpose(Y_rain_mm, tuple([1, 0, 2, 3, 4])) #
    # Y_rain_mm = np.transpose(Y_rain_mm, tuple([0, 2, 1, 3, 4]))
    # modified_shape = np.shape(Y_rain_mm)
    # modified_shape = list(modified_shape)
    # modified_shape[2] = 3
    # modified_shape = tuple(modified_shape)
    # copy = np.zeros(shape=modified_shape)
    # np.copyto(copy, Y_rain_mm)
    # Y_rain_mm = copy

    # print(Y_rain_mm[0][1])

    exit(0)

    return Y_rain_mm[lbound:rbound]

def get_numerical_data_and_validity_ndarray(lbound: int, rbound: int):
    with np.load('data/processed/supervised.npz', allow_pickle=False) as data:
        X_numerical = data['X_numerical']
        numerical_valid = data['X_num_mask']
    mask_inclusive_reshape = tuple(np.shape(X_numerical)[:1] + tuple([2]) + np.shape(X_numerical)[1:])
    numerical_validity_wrapper = np.empty(mask_inclusive_reshape)
    #print(np.shape(numerical_validity_wrapper))

    print("Numerical data copy")
    for x in range(np.shape(numerical_validity_wrapper)[0]):
        np.copyto(numerical_validity_wrapper[x][0], X_numerical[x])
    #print(numerical_validity_wrapper[0][0][0])

    print("Numerical validity copy")
    for x in range(np.shape(numerical_validity_wrapper)[0]):
        np.copyto(numerical_validity_wrapper[x][1], numerical_valid[x])
    #print(numerical_validity_wrapper[0][1][0])
    print("Done")
    numerical_validity_wrapper = np.transpose(numerical_validity_wrapper, (0, 3, 1, 2, 4, 5))
    return numerical_validity_wrapper[lbound:rbound]

def get_rain_data_and_validity_ndarray(lbound: int, rbound:int):
    with np.load('data/processed/supervised.npz', allow_pickle=False) as data:
        Y_rain_mm = data['Y_rain_mm']
        rain_valid = data['Y_rain_mask']
    mask_inclusive_reshape = tuple(np.shape(Y_rain_mm)[:1] + tuple([2]) + np.shape(Y_rain_mm)[1:])
    rain_validity_wrapper = np.empty(mask_inclusive_reshape)

    print("Rain data copy")
    for x in range(np.shape(rain_validity_wrapper)[0]):
        np.copyto(rain_validity_wrapper[x][0], Y_rain_mm[x])


    print("rain validity copy")
    for x in range(np.shape(rain_validity_wrapper)[0]):
        np.copyto(rain_validity_wrapper[x][1], rain_valid[x])
    print("Done")
    rain_validity_wrapper = np.transpose(rain_validity_wrapper, (0, 2, 1, 3, 4))
    rain_validity_wrapper = rain_validity_wrapper[np.newaxis, ...]
    rain_validity_wrapper = np.repeat(rain_validity_wrapper, 3, axis=0)
    rain_validity_wrapper = np.transpose(rain_validity_wrapper, (1, 2, 3, 0, 4, 5))
    
    return rain_validity_wrapper[lbound:rbound]