import math
import numpy as np


def gaussian(x, mean, std):
    e = math.e**(-((x-mean)**2)/(2*std**2))
    result = (1 / (std * math.sqrt(2*math.pi))) * e
    return round(result, 2)

def gaussian_list(numbers, mean, std):
    results = []
    for num in numbers:
        e = math.e**(-((num-mean)**2)/(2*std**2))
        result = round((1 / (std * math.sqrt(2*math.pi))) * e, 2)
        results.append(result)
    
    return results

def gaussian_np(array, mean, std):
    results = np.array([])
    for num in array:
        e = np.e**(-((num-mean)**2)/(2*std**2))
        result = np.round((1 / (std * np.sqrt(2*np.pi))) * e, 2)
        results = np.append(results, result)
    
    
    return results


if __name__ == '__main__':
    array = np.array([-0.1, 0.01, -3])
    mean = 0
    std = 3
    
    results = gaussian_np(array, mean, std)
    # assert (results == np.array([0.4 , 0.39 , 0. , 0.05 , 0. ])).all()
    print(results)
    