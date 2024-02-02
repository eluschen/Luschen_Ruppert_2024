# Script containing useful functions I use

# Emily Luschen - emily.w.luschen-1@ou.edu
# James Ruppert - jruppert@ou.edu
# 5/19/23

import numpy as np
from netCDF4 import Dataset
from wrf import getvar, disable_xarray
import matplotlib.pyplot as plt
from matplotlib import ticker

# Read in variable from nc file
def var_read(infile, varname):
    ncfile = Dataset(infile)
    var = ncfile.variables[varname][...]
    ncfile.close()
    var = np.squeeze(var)
    return var

# Like var_read but can choose specific level
def var_read_level(infile, varname, level):
    ncfile = Dataset(infile)
    var = ncfile.variables[varname][:,level,:,:]
    ncfile.close()
    var = np.squeeze(var)
    return var

# Like var_read but for wrf
def var_wrfread(infile, varname):
    disable_xarray()
    ncfile = Dataset(infile)
    var = getvar(ncfile, varname)
    ncfile.close()
    var = np.squeeze(var)
    return var

# Mask domain edges of variable
def mask_edges(array):
    # Returns a masked array with edges masked
    # Last dimensions of input array must be x1,x2
    #   It is otherwise versatile
    buffer=80
    array = np.ma.array(array, mask=False, copy=False)
    array[...,0:buffer,:]=np.ma.masked
    array[...,-buffer:,:]=np.ma.masked
    array[...,:,0:buffer]=np.ma.masked
    array[...,:,-buffer:]=np.ma.masked
    # array = np.ma.filled(array, fill_value=np.nan)
    return array

# combines var_read and mask_edges
def var_read_edges(infile, varname):
    ncfile = Dataset(infile)
    var = ncfile.variables[varname][...]
    ncfile.close()
    var = np.squeeze(var)
    var = mask_edges(var)
    return var

# same as var_read_edges but for wrf
def var_wrfread_edges(infile, varname):
    disable_xarray()
    ncfile = Dataset(infile)
    var = getvar(ncfile, varname)
    ncfile.close()
    var = np.squeeze(var)
    var = mask_edges(var)
    return var

# like var_read_edges but can select time
def var_read_edges_time(infile, varname, t0, t1):
    ncfile = Dataset(infile)
    var = ncfile.variables[varname][t0:t1,...]
    ncfile.close()
    var = np.squeeze(var)
    var = mask_edges(var)
    return var


# Plotting Functions

# CRF profiles
def plot_crfprofiles_class(var_profile, precip_list, var_conv, var_strat, xlabel, pres):
    
    fig = plt.figure(figsize=(12,6),facecolor='white', dpi=300)

    # Profile 1
    ax = fig.add_subplot(131)
    ax.plot(var_profile[0], pres, label=precip_list[0], color='teal',linewidth=1.5)
    ax.plot(var_profile[1], pres, label=precip_list[1], color='plum', linewidth=1.5)
    ax.plot(var_profile[2], pres, label=precip_list[2], color='darkorange', linewidth=1.5)
    ax.plot(var_conv, pres, label='Avg', linestyle='--', color='black', linewidth=1.5)
    ax.set_title('d) Deep+Congestus+Shallow', size=14, weight='bold')
    plt.legend(loc='upper right', fontsize=12)
    plt.gca().invert_yaxis()
    ax.axvline(0, color='black', alpha=0.5)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ytick_loc=np.arange(1000,0,-100)
    plt.yticks(ticks=ytick_loc, size=12)
    plt.ylim(1000, 100)
    xticks=np.arange(-8,9,2)
    plt.xticks(ticks=xticks,size=12)
    plt.xlabel(xlabel, size=14, weight='bold')
    plt.ylabel('Pressure [hPa]', size=14, weight='bold')

    # Profile 2
    ax = fig.add_subplot(132)
    ax.plot(var_profile[3], pres, label=precip_list[3], color='goldenrod', linewidth=1.5)
    ax.plot(var_profile[4], pres, label=precip_list[4], color='cornflowerblue', linewidth=1.5)
    ax.plot(var_strat, pres, label='Avg.', linestyle='--', color='black', linewidth=1.5)
    ax.set_title('e) Strat+Anvil', size=14, weight='bold')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.legend(loc='upper right', fontsize=12)
    plt.gca().invert_yaxis()
    ax.axvline(0, color='black', alpha=0.5)
    plt.ylim(1000,100)
    plt.yticks(ticks=ytick_loc, size=12)
    plt.ylim(1000, 100)
    plt.xticks(ticks=xticks,size=12)
    plt.xlabel(xlabel, size=14, weight='bold')

     # Profile 3
    ax = fig.add_subplot(133)
    ax.plot(var_conv, pres, label='Conv. Avg.', color='red')
    ax.plot(var_strat, pres, label='Strat.+Anvil Avg.', color='blue')
    ax.set_title('f) Averages', size=16, weight='bold')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.legend(loc='upper right', fontsize=12)
    plt.gca().invert_yaxis()
    ax.axvline(0, color='black', alpha=0.5)
    plt.ylim(1000,100)
    plt.yticks(ticks=ytick_loc, size=12)
    plt.ylim(1000, 100)
    plt.xticks(ticks=xticks,size=12)
    plt.xlabel(xlabel, size=14, weight='bold')

# W profiles
def plot_wprofiles_class(var_profile, precip_list, var_conv, var_strat, var_profile_og, precip_list_og, var_conv_og, var_strat_og, xlabel, pres):
    
    fig = plt.figure(figsize=(12,6),facecolor='white', dpi=300)

    # Profile 1
    ax = fig.add_subplot(131)
    ax.plot(var_profile[0], pres, label=precip_list[0], color='teal',linewidth=1.5)
    ax.plot(var_profile[1], pres, label=precip_list[1], color='plum', linewidth=1.5)
    ax.plot(var_profile[2], pres, label=precip_list[2], color='darkorange', linewidth=1.5)
    ax.plot(var_conv, pres, label='Avg.', color='black', linewidth=1.5)
    ax.plot(var_profile_og[0], pres, label='Refl. Conv.', linestyle='--', color='black', linewidth=1.5)
    ax.set_title('a) Convective', size=14, weight='bold')
    plt.legend(loc='upper right', fontsize=12)
    plt.gca().invert_yaxis()
    ax.axvline(0, color='black', alpha=0.5)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ytick_loc=np.arange(1000,0,-100)
    plt.yticks(ticks=ytick_loc, size=12)
    plt.ylim(1000, 100)
    xticks=np.arange(-0.2,1.2,.2)
    plt.xticks(ticks=xticks,size=12)
    # plt.xticks(size=12)
    plt.xlabel(xlabel, size=14, weight='bold')
    plt.ylabel('Pressure [hPa]', size=14, weight='bold')

    # Profile 2
    ax = fig.add_subplot(132)
    # print(var_strat)
    ax.plot(var_profile[3], pres, label=precip_list[3], color='goldenrod', linewidth=1.5)
    ax.plot(var_profile[4], pres, label=precip_list[4], color='cornflowerblue', linewidth=1.5)
    ax.plot(var_profile_og[1], pres, label='Refl. Strat.', linestyle='--', color='goldenrod', linewidth=1.5)
    ax.plot(var_profile_og[2], pres, label='Refl. Anvil', linestyle='--', color='cornflowerblue', linewidth=1.5)
    # ax.plot(var_strat, pres, label='Avg.', color='black', linewidth=1.5)
    # ax.plot(var_strat_og, pres, label='Refl. Avg.', linestyle='--', color='black', linewidth=1.5)
    ax.set_title('b) Strat+Anvil', size=14, weight='bold')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.legend(loc='upper left', fontsize=12)
    plt.gca().invert_yaxis()
    ax.axvline(0, color='black', alpha=0.5)
    plt.ylim(1000,100)
    plt.yticks(ticks=ytick_loc, size=12)
    plt.ylim(1000, 100)
    xticks2=np.arange(-0.2,0.3,0.1)
    plt.xticks(ticks=xticks2,size=12)
    plt.xticks(size=12)
    plt.xlabel(xlabel, size=14, weight='bold')

     # Profile 3
    ax = fig.add_subplot(133)
    ax.plot(var_conv, pres, label='Conv.', color='red')
    ax.plot(var_strat, pres, label='Strat.+Anvil', color='blue')
    ax.plot(var_conv_og, pres, label='Refl. Conv.', linestyle='--', color='red')
    ax.plot(var_strat_og, pres, label='Refl. Strat.+Anvil', linestyle='--', color='blue')
    ax.set_title('c) Averages', size=16, weight='bold')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.legend(loc='upper right', fontsize=12)
    plt.gca().invert_yaxis()
    ax.axvline(0, color='black', alpha=0.5)
    plt.ylim(1000,100)
    plt.yticks(ticks=ytick_loc, size=12)
    plt.ylim(1000, 100)
    plt.xticks(ticks=xticks,size=12)
    plt.xticks(size=12)
    plt.xlabel(xlabel, size=14, weight='bold')
