import numpy as np
import pandas as pd
from shutil import unpack_archive
import matplotlib.pyplot as plt
from matplotlib import colors
import time as tt
import os
import re

header = ['x','y','dens','frho','T','u.x','uf.x','Ilaser','IlaserAbsorp',
          'n0','ne','ni1','ni2','x0','xe','xi','xi2',
          'alphaIBen','alphaIBei','Level']

def get_header():
    return header

def get_data(results_directory,loud=True):
    """
    This function returns the results of a simulation as a list
    of Pandas DataFrames. Each element of the list corresponds
    to the results at a certain time, starting at 0 and increasing.
    Just call the function with the path of the results directory.
    """
    
    dr = results_directory
    
    file_list = []
    for f in os.listdir(dr):
        if f.endswith('_ns.dat'):
            file_list.append(f)
    l = len(file_list)
    if loud:
        print('Found',l,'result files.')
    
    data = []
    if loud:
        print('Building your list now, please hold.')
    for i in range(len(file_list)):
        path = dr+'/'+file_list[i]
        d = np.loadtxt(path)
        ddf = pd.DataFrame(d,columns=header)
        result = re.search('_(.*)_', file_list[i])
        t = float(result[1])*1e-9
        ddf['t'] = t
        data.append(ddf)
        if i == int(l/2):
            print('Halfway done...')
    data.sort(key = lambda x: x['t'][0])
    if loud:
        print('Done.')
    return data

def get_time(data):
    time=np.zeros(len(data))
    for i in range(len(data)):
        time[i]=data[i]['t'][0]
    return time

def find_nearest(array, value, return_index=False):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if return_index:
        return idx
    else:
        return array[idx]

def get_time_data(data,key,d,loud=True, return_nearest=False):
    if key not in header:
        print('You have to pick from the list:',header)
        
    time = get_time(data)        
    X=np.zeros(len(time))
    if loud:
        print('Slicing the data now...')
    
    for i in range(len(data)):
        if i == 1:
            timer0 = tt.time()
            
        nearest_index = find_nearest(data[i]['x'],d,return_index=True)
        X[i]=data[i][key][nearest_index]
        
        if return_nearest:
            print('Nearest x was {} m'.format(data[i]['x'][nearest_index]))
        
        if i == 1:
            timer1 = tt.time()
            timer = timer1-timer0
            #if loud:
                #print('I think this slice will probably take about',timer*len(data)/60,'minutes.')
    timer_final = (tt.time()-timer0)
    if loud:
        print('This slice took {:.2f} seconds so consider that for future slices.'.format(timer_final))
        
    return time,X

def get_x_data(data,key,t,loud=True,return_nearest=False):
    if key not in header:
        print('You have to pick from the list:',header)
    time = get_time(data)
    data_idx = find_nearest(time,t, return_index=True)
    
    df = pd.DataFrame()
    df['x'] = data[data_idx]['x']
    df['key'] = data[data_idx][key]
    df=df.sort_values(by='x')
    
    if return_nearest:
        print('Nearest time was {:.2f} ns'.format(time[data_idx]*1e9))
    
    return df['x'],df['key']
    
def create_map_data(data,x_array, y_array, key):
    mapdata = np.zeros((len(y_array),len(x_array)))
    
    for i in range(len(y_array)):
        mapdata[i,:] = get_time_data(data,key,y_array[i],loud=False)[1]
    
    return mapdata

def plot_map(x_array, y_array, mapdata, levels=None, log=False):
    if levels is not None:
        CF = plt.contourf(x_array,y_array,mapdata,levels=levels)

    if log:
        CF = plt.contourf(x_array,y_array,mapdata,levels=levels,norm=colors.LogNorm())

    else:
        CF = plt.contourf(x_array,y_array,mapdata)
    plt.colorbar(CF,ticks=levels)
            
            
    
    
    
    
    
    