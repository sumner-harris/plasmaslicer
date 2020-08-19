import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
import os
import re
from scipy.optimize import curve_fit

def load_single(path_to_file, laser_energy=None):
    E = laser_energy
    df = pd.read_csv(path_to_file,skiprows=21,header=1) #12 is the minimum number of rows to skip
    df = df.drop(['X_Value'],axis=1)
    df = df.drop(['Comment'],axis=1)
    df = df.rename(columns = {'Untitled':'Time'})
    df['Ion Mass'] = df['Ion Mass'][0]
    df['Distance to Target'] = df['Distance to Target'][0]
    df['Bias Voltage'] = df['Bias Voltage'][0]
    df['Laser Energy'] = E

    return df
    
def IV_load_data(directory,laser_energy):
    
    file_list = [os.path.join(directory,f) for f in os.listdir(directory) if f.endswith('.lvm')]
    
    data = []
    for i in range(len(file_list)):

        E = laser_energy
        df = pd.read_csv(file_list[i],skiprows=21,header=1) #12 is the minimum number of rows to skip
        df = df.drop(['X_Value'],axis=1)
        df = df.drop(['Comment'],axis=1)
        df = df.rename(columns = {'Untitled':'Time'})
        df['Ion Mass'] = df['Ion Mass'][0]
        df['Distance to Target'] = df['Distance to Target'][0]
        df['Bias Voltage'] = df['Bias Voltage'][0]
        df['Laser Energy'] = E
        data.append(df)

    #This line sorts the table in the list according to laser energy
    data.sort(key = lambda x: x['Bias Voltage'][0])
    
    return data

def PAT_load_data(directory, search_format):
    """
    The search_format variable is a string that looks for the laser energy in the file name.
    It needs to be written like a search for python regular expressions like: 'FWO-(.*)mJ'
    which will take everything between FWO- and mJ and save it as the energy for that file.
    """
    
    file_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.lvm')]
    data = []

    for i in range(len(file_list)):

        result = re.search(search_format, file_list[i])
        E = float(result.group(1).replace('_','.'))
        df = pd.read_csv(file_list[i],skiprows=21,header=1) #12 is the minimum number of rows to skip
        df = df.drop(['X_Value'],axis=1)
        df = df.drop(['Comment'],axis=1)
        df = df.rename(columns = {'Untitled':'Time'})
        df['Ion Mass'] = df['Ion Mass'][0]
        df['Distance to Target'] = df['Distance to Target'][0]
        df['Bias Voltage'] = df['Bias Voltage'][0]
        df['Laser Energy'] = E
        data.append(df)
    #This line sorts the table in the list according to laser energy
    data.sort(key = lambda x: x['Laser Energy'][0])

    return data

def pulse_set_zero(df):
    first_microsecond = df['Voltage'][df['Time']<=1e-6]
    average = first_microsecond.mean()
    offset_voltage = df['Voltage']-average
    probe_area = ((df['Voltage']/10*df['Time']**3)/(df['Ion Mass']*df['Distance to Target']**2*1.603e-19*6.242e18*df['dN/dE'])).mean()
    #print(probe_area)
    df['Offset_Applied'] = offset_voltage
    df['dN/dE_offset'] = (offset_voltage/10*df['Time']**3)/(df['Ion Mass']*df['Distance to Target']**2*1.603e-19*6.242e18*probe_area)
    
    return df

def dataset_set_zero(data_list):
    new_data = []
    for i in range(len(data_list)):
        df = pulse_set_zero(data_list[i])
        new_data.append(df)

    return(new_data)

def find_nearest(array, value, return_index=False):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if return_index:
        return idx
    else:
        return array[idx]

def get_biases(data_list):
    biases = np.zeros(len(data_list))
    for i in range(len(data_list)):
        biases[i]=data_list[i]['Bias Voltage'][0]
    return biases

def get_laser_energies(data_list):
    laser_energy = np.zeros(len(data_list))

    for i in range(len(data_list)):
        laser_energy[i] = data_list[i]['Laser Energy'][0] #Pulls the value of laser energy
    return laser_energy

def get_IV(data_list,time,resistance, offset=True):
    t_index = find_nearest(data_list[0]['Time'],time, return_index = True)
    biases = get_biases(data_list)
    I_V = np.zeros(len(data_list))
    
    if offset:
        for i in range(len(data_list)):
            I_V[i]=-data_list[i]['Offset_Applied'][t_index]/resistance
    else:
        for i in range(len(data_list)):
            I_V[i]=-data_list[i]['Voltage'][t_index]/resistance        
    return biases, I_V

def calculate_dNdE(df,resistance, ion_mass=None, distance_to_target=None,
                   probe_area=None, offset=True):
    e = 1.603e-19 #electron charge
    J2eV = 6.242e18 #convert J to eV
    if ion_mass == None:
        ion_mass = df['Ion Mass']
    if distance_to_target == None:
        distance_to_target=df['Distance to Target']
    if probe_area == None:
        probe_area = ((df['Voltage']/10*df['Time']**3)/\
                      (ion_mass*distance_to_target**2*e*J2eV*df['dN/dE'])).mean()
        
    energy = 0.5*ion_mass*(distance_to_target/df['Time'])**2*J2eV
    
    dNdE = (df['Offset_Applied']/resistance*df['Time']**3)/(ion_mass*distance_to_target**2*e*J2eV*probe_area)
    if offset == False:
        dNdE = (df['Voltage']/resistance*df['Time']**3)/(ion_mass*distance_to_target**2*e*J2eV*probe_area)
    return energy, dNdE

def integrate_charge(data_list,resistance, offset=True):
    integrated_currents = np.zeros(len(data_list))
    if offset == True:
        offset_list = dataset_set_zero(data_list).copy()
        for i in range(len(offset_list)):
            integrated_currents[i] = np.trapz(offset_list[i]['Offset_Applied']/resistance,offset_list[i]['Time'])
    else:
        for i in range(len(data_list)):
            #integrate the current with np.trapz method
            integrated_currents[i] = np.trapz(data_list[i]['Voltage']/resistance,data_list[i]['Time'])
    return integrated_currents

def quick_plot(dataframe, offset=False):
    time = dataframe['Time']*1e6 # multiply by 1e6 to convert to microseconds
    if offset:
        voltage = dataframe['Offset_Applied']
    else:
        voltage = dataframe['Voltage']

    kinetic_energy = dataframe['Energy']
    dNdE = dataframe['dN/dE']
    
    upper_x_limit = find_nearest(dNdE,1e11, True)
    #print(upper_x_limit)

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].plot(time, voltage,'b')
    ax[0].set_xlabel('time ($\mu$s)')
    ax[0].set_ylabel('probe voltage (V)')

    ax[1].semilogy(kinetic_energy, dNdE, 'b')
    ax[1].set_ylabel('dN/dE (m$^{-2}$ eV${-1}$)')
    ax[1].set_xlabel('Kinetic Energy (eV)')
    ax[1].set_xlim(0,kinetic_energy[upper_x_limit])
    ax[1].set_ylim(bottom=1e11)
    plt.show()

def calculate_threshold(laser_energy, integrated_current, upper_line_bounds, lower_line_bounds,
                        scale):
    if scale not in ['linear','loglin','loglog']:
        raise ValueError('Must choose scale from linear, loglin, or loglog scale.')
    
    upper_line_limits = (laser_energy>=upper_line_bounds[0])&(laser_energy<=upper_line_bounds[1])
    lower_line_limits = (laser_energy>=lower_line_bounds[0])&(laser_energy<=lower_line_bounds[1])
    not_used = (laser_energy>=lower_line_bounds[1])&(laser_energy<=upper_line_bounds[0])
    
    x1 = laser_energy[upper_line_limits]
    x2 = laser_energy[lower_line_limits]
    y1 = integrated_current[upper_line_limits]
    y2 = integrated_current[lower_line_limits]
    #print(x1.shape, x2.shape, y1.shape, y2.shape)
    
    if scale =='loglin':
        y2_masked = np.ma.masked_array(y2,y2<=0)
        x2_masked = np.ma.masked_where(np.ma.getmask(y2_masked), x2)
        m1, b1, r1, p1, stderr1 = linregress(x1,np.log(y1))
        m2, b2, r2, p2, stderr2 = linregress(x2_masked,np.log(y2_masked))
        #print(m1,m2,b1,b2)
        
        plasma_thresh = (b1-b2) / (m2-m1)
        
        fig,ax = plt.subplots()

        ax.semilogy(x1, y1,'ob',alpha=0.5, label='Fitted Points')
        ax.semilogy(x2, y2,'ob',alpha=0.5)
        ax.plot(laser_energy, np.exp(laser_energy*m1+b1),'k')
        ax.plot(laser_energy, np.exp(laser_energy*m2+b2),'k')
        ax.plot(laser_energy[not_used],integrated_current[not_used],'or',alpha=0.5,label='Excluded from fit')
        ax.plot(plasma_thresh, np.exp(m1*plasma_thresh+b1),'ok')
        ax.text(0.75, 0.5, 'Plasma Absorptiom Threshold:\n{:.2f} mJ'.format(plasma_thresh),
                 horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes)
        ax.set_xlim(0,max(laser_energy))
        ax.set_ylim(min(y2_masked),max(integrated_current)*2)
        ax.set_xlabel('Laser Energy (mJ)')
        ax.set_ylabel('Integrated Charge (C/m$^2$)')
        ax.legend()
        
    if scale =='linear':
        y2_masked = np.ma.masked_array(y2,y2<=0)
        x2_masked = np.ma.masked_where(np.ma.getmask(y2_masked), x2)
        m1, b1, r1, p1, stderr1 = linregress(x1,y1)
        m2, b2, r2, p2, stderr2 = linregress(x2_masked,y2_masked)
        #print(m1,m2,b1,b2)
        
        plasma_thresh = (b1-b2) / (m2-m1)
        
        fig,ax = plt.subplots()

        ax.plot(x1, y1,'ob',alpha=0.5, label='Fitted Points')
        ax.plot(x2, y2,'ob',alpha=0.5)
        ax.plot(laser_energy, laser_energy*m1+b1,'k')
        ax.plot(laser_energy, laser_energy*m2+b2,'k')
        ax.plot(laser_energy[not_used],integrated_current[not_used],'or',alpha=0.5,label='Excluded from fit')
        ax.plot(plasma_thresh, np.exp(m1*plasma_thresh+b1),'ok')
        ax.text(0.75, 0.5, 'Plasma Absorptiom Threshold:\n{:.2f} mJ'.format(plasma_thresh),
                 horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes)
        ax.set_xlim(0,max(laser_energy))
        ax.set_ylim(min(y2_masked),max(integrated_current))
        ax.set_xlabel('Laser Energy (mJ)')
        ax.set_ylabel('Integrated Charge (C/m$^2$)')
        ax.legend()
        
    if scale == 'loglog':
        y2_masked = np.ma.masked_array(y2,y2<=0)
        x2_masked = np.ma.masked_where(np.ma.getmask(y2_masked), x2)
        m1, b1, r1, p1, stderr1 = linregress(np.log(x1),np.log(y1))
        m2, b2, r2, p2, stderr2 = linregress(np.log(x2_masked),np.log(y2_masked))
        #print(m1,m2,b1,b2)
        
        plasma_thresh = (b1-b2) / (m2-m1)
        
        fig,ax = plt.subplots()

        ax.loglog(x1, y1,'ob',alpha=0.5, label='Fitted Points')
        ax.plot(x2, y2,'ob',alpha=0.5)
        ax.plot(laser_energy, np.exp(b1)*laser_energy**(m1),'k')
        ax.plot(laser_energy, np.exp(b2)*laser_energy**(m2),'k')
        ax.plot(laser_energy[not_used],integrated_current[not_used],'or',alpha=0.5,label='Excluded from fit')
        #ax.plot(plasma_thresh, np.exp(b1)*plasma_thresh**m1,'ok')
        ax.text(0.75, 0.5, 'Plasma Absorptiom Threshold:\n{:.2f} mJ'.format(plasma_thresh),
                 horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes)
        ax.set_xlim(0,max(laser_energy))
        ax.set_ylim(min(y2_masked),max(integrated_current))
        ax.set_xlabel('Laser Energy (mJ)')
        ax.set_ylabel('Integrated Charge (C/m$^2$)')
        ax.legend()
    
    if m2 <=0 or np.isnan(m2):
        raise ValueError('The lower slope is NaN or negative, increase the lower energy bound')
        
    print('Plasma absorption threshold = {:.2f}'.format(plasma_thresh))
    print('Fit details:')
    print('Lower line: m = {:.5e}, b = {:.5e}'.format(m2, b2))
    print('Upper line: m = {:.5e}, b = {:.5e}'.format(m1, b1))
    results = {'Theshold': plasma_thresh, 'm_upper':m1, 'b_upper':b1 , 'm_lower':m2, 'b_lower':b2}
    return results
