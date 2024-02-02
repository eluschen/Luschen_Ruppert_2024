#!/usr/bin/env python
# coding: utf-8

# ### Python Script for CRF and W Profiles Across All Time and Members
# 
# Emily Luschen
# Emily.W.Luschen-1@ou.edu
# James Ruppert  
# jruppert@ou.edu  
# 11/15/23

# import packages
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import master_functions as mf

#### CHANGE THESE THINGS #####
nmem = 10 # number of members evaluating
t0 = 12 # start time
t1 = 36 # end time
pres = np.arange(50,1025,25) # to get pressure levels
istorm = 'haiyan' # Switch cloud class ncfile below based on storm!!
###############################

# setup variables
membs = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'] # members list
time_range = np.arange(t0,t1,1)
ntime = len(time_range)
precip_list = ['Deep', 'Congestus', 'Shallow', 'Stratiform', 'Anvil'] # new classifcation list
nclass = len(precip_list)
nz = len(pres)
precip_list_og = ['Convective', 'Stratiform', 'Anvil'] # reflectivity class list
nclass_og = len(precip_list_og)

# read in new cloud classification
ncfile = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/cloud_class/precip_class_ctl.nc' # haiyan
# ncfile = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/cloud_class/precip_class_ctl_maria.nc' #maria
with Dataset(ncfile, 'r') as nc:
    c_type = nc.variables['type'][:,t0:t1,:,:] # memb x time x lat x lon; 1=deep convective, 2=congestus, 3=shallow convective, 4=stratiform, 5=anvil, 0=nonraining

## CRF PROFILES

# initialize arrays
crf_mem_type = np.empty((nmem,nclass,nz)) # mem x cloud type x level
conv_mem = np.empty((nmem,nz)) # mem x level
strat_mem = np.empty((nmem,nz)) # mem x level

for m in range(nmem):
    # read in files to calculate CRF
    file = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/tc_ens/'+istorm+'/memb_'+membs[m]+'/ctl/post/d02/RTHRATLW_HiRes.nc'
    lw = mf.var_read_edges_time(file,'RTHRATLW',t0,t1)*3600*24 # K/s --> K/d
    file = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/tc_ens/'+istorm+'/memb_'+membs[m]+'/ctl/post/d02/RTHRATLWC_HiRes.nc'
    lwc = mf.var_read_edges_time(file,'RTHRATLWC',t0, t1)*3600*24 # K/s --> K/d
    file = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/tc_ens/'+istorm+'/memb_'+membs[m]+'/ctl/post/d02/RTHRATSW_HiRes.nc'
    sw = mf.var_read_edges_time(file, 'RTHRATSW',t0,t1)*3600*24 # K/s --> K/d
    file = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/tc_ens/'+istorm+'/memb_'+membs[m]+'/ctl/post/d02/RTHRATSWC_HiRes.nc'
    swc = mf.var_read_edges_time(file, 'RTHRATSWC',t0,t1)*3600*24 # K/s --> K/d
    crf = sw + lw - lwc - swc # time x level x lat x lon
    del sw
    del lw
    del lwc
    del swc
    print('CRF calculated')


    c_type_memb = c_type[m] # choose member for cloud classification

    # average CRF across time and space as a function of cloud type
    for t in range(nclass): # loop through cloud type
        ind = (c_type_memb == t+1).nonzero()
        crf_mem_type[m,t,:] = np.ma.mean(crf[ind[0],:,ind[1],ind[2]], axis=0)

    # avg convective and stratiform+anvil CRF across time and space
    ind = ((c_type_memb > 0) & (c_type_memb < 4)).nonzero()
    conv_mem[m,:] = np.ma.mean(crf[ind[0],:,ind[1],ind[2]], axis=0) # convective total
    ind = (c_type_memb > 3).nonzero()
    strat_mem[m,:]= np.ma.mean(crf[ind[0],:,ind[1],ind[2]], axis=0) # stratiform total

    del crf

    print(membs[m])

# average across members
crf_type_prof = np.nanmean(crf_mem_type, axis=0)
conv_prof = np.nanmean(conv_mem, axis=0)
strat_prof = np.nanmean(strat_mem, axis=0)

del crf_mem_type
del conv_mem
del strat_mem

# Plot!
pres_plot = np.flip(pres)
print(pres_plot.shape)
xlabel = 'CRF [K/day]'

mf.plot_crfprofiles_class(crf_type_prof, precip_list, conv_prof, strat_prof, xlabel, pres_plot)
plt.savefig('/home/eluschen/figures/profiles/crf_profile_'+str(nmem)+'mem_time_'+str(t0)+'_'+str(t1)+'_'+istorm+'.png')

del crf_type_prof
del conv_prof
del strat_prof

print('CRF Profiles Complete')

## W Profiles

# initialize arrays => 'og' references reflectivity based classification or Rogers (2010)
w_mem_type = np.empty((nmem,nclass,nz)) # mem x cloud type x level
w_mem_type_og = np.empty((nmem,nclass_og,nz)) # mem x cloud type x level
conv_mem = np.empty((nmem,nz)) # mem x level
strat_mem = np.empty((nmem,nz)) # mem x level
conv_mem_og = np.empty((nmem,nz)) # mem x level
strat_mem_og = np.empty((nmem,nz)) # mem x level

for m in range(nmem):
    # read in w and og classification
    file = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/tc_ens/'+istorm+'/memb_'+membs[m]+'/ctl/post/d02/W_HiRes.nc'
    w = mf.var_read_edges_time(file,'W',t0,t1) # time x level x lat x lon
    pclass_trad = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/tc_ens/'+istorm+'/memb_'+membs[m]+'/ctl/post/d02/strat.nc'
    c_type_og = mf.var_read_edges_time(pclass_trad, 'strat', t0, t1) # time x lat x lon; 1=convective, 2=stratiform, 3=anvil, 0=nonraining

    # New Classification
    c_type_memb = c_type[m] # new class type by member

    # average W across time and space as a function of cloud type
    for t in range(nclass): # loop through cloud type
        ind = (c_type_memb == t+1).nonzero()
        w_mem_type[m,t,:] = np.ma.mean(w[ind[0],:,ind[1],ind[2]], axis=0) # memb x type x level

    # avg convective and stratiform+anvil W across time and space
    ind = ((c_type_memb > 0) & (c_type_memb < 4)).nonzero() # convective index
    conv_mem[m,:] = np.ma.mean(w[ind[0],:,ind[1],ind[2]], axis=0) # convective total
    ind = (c_type_memb > 3).nonzero() # stratiform + anvil index
    strat_mem[m,:]= np.ma.mean(w[ind[0],:,ind[1],ind[2]], axis=0) # stratiform total

    print('New Classifcation Done')

    # Original/Traditional Classification
    for c in range(nclass_og): # loop through cloud type
        ind = (c_type_og == c+1).nonzero()
        w_mem_type_og[m,c,:] = np.ma.mean(w[ind[0],:,ind[1],ind[2]], axis=0) # memb x type x level

    # avg convective and stratiform+anvil W across time and space
    ind = (c_type_og == 1).nonzero() # convective index
    conv_mem_og[m,:] = np.ma.mean(w[ind[0],:,ind[1],ind[2]], axis=0) # convective total
    ind = (c_type_og > 1).nonzero() # stratiform + anvil index
    strat_mem_og[m,:]= np.ma.mean(w[ind[0],:,ind[1],ind[2]], axis=0) # stratiform total

    print('Reflectivity Classifcation Done')

    del w
    del c_type_og

    print(membs[m])

#average across members for new classification
w_type_prof = np.nanmean(w_mem_type, axis=0)
conv_prof = np.nanmean(conv_mem, axis=0)
strat_prof = np.nanmean(strat_mem, axis=0)

#average across members for reflectivity classification
w_type_prof_og = np.nanmean(w_mem_type_og, axis=0)
conv_prof_og = np.nanmean(conv_mem_og, axis=0)
strat_prof_og = np.nanmean(strat_mem_og, axis=0)

# delete things for memory
del w_mem_type
del conv_mem
del strat_mem
del w_mem_type_og
del conv_mem_og
del strat_mem_og

# Plot!!
pres_plot = np.flip(pres)
print(pres_plot.shape)
xlabel = 'w [m/s]'

mf.plot_wprofiles_class(w_type_prof, precip_list, conv_prof, strat_prof, w_type_prof_og, precip_list_og, conv_prof_og, strat_prof_og, xlabel, pres_plot)
plt.savefig('/home/eluschen/figures/profiles/w_profile_'+str(nmem)+'mem_time_'+str(t0)+'_'+str(t1)+'_'+istorm+'.png')

print('W Profiles Complete')