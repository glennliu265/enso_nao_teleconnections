#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform Lag Regression for ENSO index and selected variable (anomalized and detrended)

Created on Thu Oct  9 16:01:18 2025

@author: gliu
"""

import sys
import time
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy import stats
import xarray as xr
import sys
import tqdm

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
import cartopy.feature as cfeature


#%% Functions

def swap_rename(ds,chkvar,newvar):
    # Swap/replace names of variables in DataArray
    if chkvar in list(ds.coords):
        print("Renaming [%s] to [%s]" % (chkvar,newvar))
        ds = ds.rename({chkvar:newvar})
    return ds

def match_time_month(var_in,ts_in,timename='time'):
    # Crops the start and end times for var_in and ts_in (xr.DataArrays/Datasets)
    # Note works for datetime64[ns] format in xr.DataArray
    # See ensobase/calculate_enso_response.py for working example
    
    if len(var_in[timename]) != len(ts_in[timename]): # Check if they match
        
        # Warning: Only checking Year and Date
        vstart = str(np.array((var_in[timename].data[0])))[:7]
        tstart = str(np.array((ts_in[timename].data[0])))[:7]
        
        if vstart != tstart:
            print("Start time (v1=%s,v2=%s) does not match..." % (vstart,tstart))
            if vstart > tstart:
                print("Cropping to start from %s" % vstart)
                ts_in = ts_in.sel( 
                    {timename : slice(vstart+"-01",None)}
                    )
            elif vstart < tstart:
                print("Cropping to start from %s" % tstart)
                var_in = var_in.sel(
                    {timename : slice(tstart+"-01",None)}
                    )
        
        vend = str(np.array((var_in[timename].data[-1])))[:7]
        tend = str(np.array((ts_in[timename].data[-1])))[:7]
        
        
        if vend != tend:
            
            print("End times (v1=%s,v2=%s) does not match..." % (vend,tend))
            
            if vend > tend:
                print("\nCropping to end at %s" % tend)
                var_in = var_in.sel(
                    {timename : slice(None,tend+"-31")}
                    )
            elif vend < tend:
                print("\nCropping to end at %s" % vend)
                ts_in = ts_in.sel(
                    {timename : slice(None,vend+"-31")}
                    )
                
        print(len(var_in[timename]) == len(ts_in[timename]))  
    return var_in,ts_in

def regress_ttest(in_var,in_ts,dof=None,p=0.05,tails=2,verbose=True):
    """
    Given a timeseries (in_ts) and variable (in_var), compute regression
    coefficients and perform t-test to get significance
    Note: only tested for single value DOF, need to check for map of dofs...
    h0: regression coeffs are significantly different from zero
    
    Inputs:
    -------
    invar (ARRAY: [Lon x Lat x Time])   : Input pattern to regress
    in_ts (ARRAY: [time])               : Timeseries to regress to
    dof   (NUMERIC)                     : Degrees of Freedom to use. Defaults to nt-2
    p     (NUMERIC)                     : p-value for significance testing; Default: 0.05
    tails (INT)                         : # of Tails for t-test (1 or 2); Default: 2
    
    Outputs: (all (ARRAY: [Lon x Lat] ezcept t_critval)
    --------
    regression_coeff : Map of Regression Coefficients
    intercept        : Map of Intercepts
    SSE              : Squared Sum of Errors
    se               : Residual Standard Error
    t_statistic      : T-statistic at each point
    t_critval        : Critical T-value
    sigmask          : Mask where t_statistic > t_critval
    
    """
    
    # Step (1), get needed dimensions
    nt          = in_ts.shape[0]
    nlon,nlat,_ = in_var.shape # Assume [lon x lat x time]
    invar_rs    = in_var.reshape(nlon*nlat,nt)
    
    # Step (2), Remove NaNs
    nandict     = find_nan(invar_rs,1,return_dict=True,verbose=verbose) # Sum along time in 1
    invar_rs    = nandict['cleaned_data']
    
    # Define function to replace NaN
    def replace(x):
        outvar = np.zeros((nlon*nlat))
        outvar[nandict['ok_indices']] = x
        return outvar.reshape(nlon,nlat)
    
    # A1. Compute the Slopes
    m,b = regress_2d(in_ts,invar_rs) # [1 x pts]
    
    # A2. Calculate SSE and residual standard error
    # https://www.geo.fu-berlin.de/en/v/soga-r/Basics-of-statistics/Hypothesis-Tests/Inferential-Methods-in-Regression-and-Correlation/Inferences-About-the-Slope/index.html
    yhat    = in_ts[None,:] * m.T  + b.T # Re-make the model
    epsilon = invar_rs - yhat # Residual
    SSE     = (epsilon**2).sum(1) # Errors are generally large along NAC
    if dof is None:
        if verbose:
            print("Using DOF len(time) - 2...")
        dof     = nt-2 # Note you can set DOF to be different here. I think 2 is just 2 parameters for linear regr
    se      = np.sqrt(SSE/ (dof)) # Residual Standard Error. 
    
    # A3. Compute the t-statistic
    rss_x = np.sqrt( np.sum( (in_ts - in_ts.mean()) **2))# Root Sum Square of x
    denom = se / rss_x
    tstat = m.squeeze() / denom
    
    # A4. Get Critical T
    ptilde  = p/tails
    critval = stats.t.ppf(1-ptilde,dof)
    if tails == 2:
        critval_lower = stats.t.ppf(ptilde,dof)
    
    # Make significance Mask
    if tails == 2:
        sigmask = (tstat > critval) | (tstat < critval_lower)
    else:
        sigmask = tstat > critval
    
    
    sigmask = replace(sigmask)
    
    # Return all values
    outdict = {}
    outdict["regression_coeff"] = replace(m.squeeze())
    outdict["intercept"] = replace(b.squeeze())
    outdict["SSE"] = replace(SSE)
    outdict["se"] = replace(se)
    outdict["t_statistic"] = replace(tstat)
    outdict["t_critval"] = critval
    outdict["sigmask"] = sigmask
    if tails == 2:
        outdict['t_critval_lower'] = critval_lower
    
    return outdict

def find_nan(data,dim,val=None,return_dict=False,verbose=True):
    """
    For a 2D array, remove any point if there is a nan in dimension [dim].
    
    Inputs:
        1) data        : 2d array, which will be summed along last dimension
        2) dim         : dimension to sum along. 0 or 1.
        3) val         : value to search for (default is NaN)
        4) return_dict : Set to True to return dictionary with clearer arguments...
    Outputs:
        1) okdata : data with nan points removed
        2) knan   : boolean array with indices of nan points
        3) okpts  : indices for non-nan points
    """
    
    # Sum along select dimension
    if len(data.shape) > 1:
        datasum = np.sum(data,axis=dim)
    else:
        datasum = data.copy()
    
    # Find non nan pts
    if val is None:
        knan  = np.isnan(datasum)
    else:
        knan  = (datasum == val)
    okpts = np.invert(knan)
    
    if len(data.shape) > 1:
        if dim == 0:
            okdata = data[:,okpts]
            clean_dim = 1
        elif dim == 1:    
            okdata = data[okpts,:]
            clean_dim = 0
    else:
        okdata = data[okpts]
    if verbose:
        print("Found %i NaN Points along axis %i." % (data.shape[clean_dim] - okdata.shape[clean_dim],clean_dim))
    if return_dict: # Return dictionary with clearer arguments
        nandict = {"cleaned_data" : okdata,
                   "nan_indices"  : knan,
                   "ok_indices"   : okpts,
                   }
        return nandict
    return okdata,knan,okpts

def regress_2d(A,B,nanwarn=1,verbose=True):
    """
    Regresses A (independent variable) onto B (dependent variable), where
    either A or B can be a timeseries [N-dimensions] or a space x time matrix 
    [N x M]. Script automatically detects this and permutes to allow for matrix
    multiplication.
    Note that if A and B are of the same size, assumes axis 1 of A will be regressed to axis 0 of B
    
    Returns the slope (beta) for each point, array of size [M]
    
    
    """
    
    # Determine if A or B is 2D and find anomalies
    bothND = False # By default, assume both A and B are not 2-D.
    # Note: need to rewrite function such that this wont be a concern...
    
    # Accounting for the fact that I dont check for equal dimensions below..
    #B = B.squeeze()
    #A = A.squeeze() Commented out below because I still need to fix some things
    # Compute using nan functions (slower)
    if np.any(np.isnan(A)) or np.any(np.isnan(B)):
        if nanwarn == 1:
            print("NaN Values Detected...")
        
        # 2D Matrix is in A [MxN]
        if len(A.shape) > len(B.shape):
            
            # Tranpose A so that A = [MxN]
            if A.shape[1] != B.shape[0]:
                A = A.T
            
            # Set axis for summing/averaging
            a_axis = 1
            b_axis = 0
            
            # Compute anomalies along appropriate axis
            Aanom = A - np.nanmean(A,axis=a_axis)[:,None]
            Banom = B - np.nanmean(B,axis=b_axis)
            
        # 2D matrix is B [N x M]
        elif len(A.shape) < len(B.shape):
            
            # Tranpose B so that it is [N x M]
            if B.shape[0] != A.shape[0]:
                B = B.T
            
            # Set axis for summing/averaging
            a_axis = 0
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.nanmean(A,axis=a_axis)
            Banom = B - np.nanmean(B,axis=b_axis)[None,:]
        

        # A is [P x N], B is [N x M]
        elif len(A.shape) == len(B.shape):
            if verbose:
                print("Note, both A and B are 2-D...")
            bothND = True
            if A.shape[1] != B.shape[0]:
                print("WARNING, Dimensions not matching...")
                print("A is %s, B is %s" % (str(A.shape),str(B.shape)))
                print("Detecting common dimension")
                # Get intersecting indices 
                intersect, ind_a, ind_b = np.intersect1d(A.shape,B.shape, return_indices=True)
                if ind_a[0] == 0: # A is [N x P]
                    A = A.T # Transpose to [P x N]
                if ind_b[0] == 1: # B is [M x N]
                    B = B.T # Transpose to [N x M]
                print("New dims: A is %s, B is %s" % (str(A.shape),str(B.shape)))
                
            # Set axis for summing/averaging
            a_axis = 1 # Assumes dim 1 of A will be regressed to dim 0 of b
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.nanmean(A,axis=a_axis,keepdims=True)#[:,None] # Anomalize w.r.t. dim 1 of A
            Banom = B - np.nanmean(B,axis=b_axis,keepdims=True)# # Anonalize w.r.t. dim 0 of B
            
        # Calculate denominator, summing over N
        Aanom2 = np.power(Aanom,2)
        denom  = np.nansum(Aanom2,axis=a_axis,keepdims=True)     # Sum along dim 1 of A (lets say this is time)
        
        # Calculate Beta
        #if 
        if len(denom.shape)==1 or not bothND: # same as both not ND
            print("Adding singleton dimension to denom")
            denom = denom[:,None]
        beta = Aanom @ Banom / denom#[:,None] # Denom is [A[mode,time]@ B[time x space]], output is [mode x pts]
        
        b = (np.nansum(B,axis=b_axis,keepdims=True) - beta * np.nansum(A,axis=a_axis,keepdims=True))/A.shape[a_axis]
        # b is [mode x pts] [or P x M]
            
    else:
        # 2D Matrix is in A [MxN]
        if len(A.shape) > len(B.shape):
            
            # Tranpose A so that A = [MxN]
            if A.shape[1] != B.shape[0]:
                A = A.T
                
            a_axis = 1
            b_axis = 0
            
            # Compute anomalies along appropriate axis
            Aanom = A - np.mean(A,axis=a_axis)[:,None]
            Banom = B - np.mean(B,axis=b_axis)
            
        # 2D matrix is B [N x M]
        elif len(A.shape) < len(B.shape):
            
            # Tranpose B so that it is [N x M]
            if B.shape[0] != A.shape[0]:
                B = B.T
            
            # Set axis for summing/averaging
            a_axis = 0
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.mean(A,axis=a_axis)
            Banom = B - np.mean(B,axis=b_axis)[None,:]
            
        # A is [P x N], B is [N x M]
        elif len(A.shape) == len(B.shape):
            if verbose:
                print("Note, both A and B are 2-D...")
            bothND = True
            if A.shape[1] != B.shape[0]:
                print("WARNING, Dimensions not matching...")
                print("A is %s, B is %s" % (str(A.shape),str(B.shape)))
                print("Detecting common dimension")
                # Get intersecting indices 
                intersect, ind_a, ind_b = np.intersect1d(A.shape,B.shape, return_indices=True)
                if ind_a[0] == 0: # A is [N x P]
                    A = A.T # Transpose to [P x N]
                if ind_b[0] == 1: # B is [M x N]
                    B = B.T # Transpose to [N x M]
                print("New dims: A is %s, B is %s" % (str(A.shape),str(B.shape)))
            
            # Set axis for summing/averaging
            a_axis = 1
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.mean(A,axis=a_axis)[:,None]
            Banom = B - np.mean(B,axis=b_axis)[None,:]

        # Calculate denominator, summing over N
        Aanom2 = np.power(Aanom,2)
        denom  = np.sum(Aanom2,axis=a_axis,keepdims=True)
        if not bothND:
            
            denom = denom[:,None] # Broadcast
            
        # Calculate Beta
        beta = Aanom @ Banom / denom
            
        if bothND:
            b = (np.sum(B,axis=b_axis)[None,:] - beta * np.sum(A,axis=a_axis)[:,None])/A.shape[a_axis]
        else:
            b = (np.sum(B,axis=b_axis) - beta * np.sum(A,axis=a_axis))/A.shape[a_axis]
    
    return beta,b

def make_encoding_dict(ds,encoding_type='zlib'):
    if type(ds) == xr.core.dataarray.DataArray:
        vname         = ds.name
        encoding_dict = {vname : {encoding_type:True}}
    else:
        keys          = list(ds.keys())
        values        = ({encoding_type:True},) * len(keys)
        encoding_dict = { k:v for (k,v) in zip(keys,values)}
    return encoding_dict


def init_tp_map(nrow=1,ncol=1,figsize=(12.5,4.5),ax=None):
    bbplot = [120, 290, -20, 20]
    fix_lon = np.hstack([np.arange(120,190,10),np.arange(-180,-60,10)])
    proj   = ccrs.PlateCarree(central_longitude=180)
    projd  = ccrs.PlateCarree()
    
    if ax is None:
        fig,axs = plt.subplots(nrow,ncol,figsize=figsize,subplot_kw={'projection':proj})
        newfig = True
    else:
        newfig = False
    if nrow != 1 or ncol != 1:
        for ax in axs.flatten():
            ax.set_extent(bbplot)
            ax     = add_coast_grid(ax,bbox=bbplot,fill_color='k',
                                        proj=ccrs.PlateCarree(),fix_lon=fix_lon,ignore_error=True)
        ax = axs
    else:
        ax = axs
        ax.set_extent(bbplot)
        ax     = add_coast_grid(ax,bbox=bbplot,fill_color='k',
                                    proj=ccrs.PlateCarree(),fix_lon=fix_lon,ignore_error=True)
    
    if newfig:
        return fig,ax
    return ax


def add_coast_grid(ax,bbox=[-180,180,-90,90],proj=None,blabels=[1,0,0,1],ignore_error=False,
                   fill_color=None,line_color='k',grid_color='gray',c_zorder=1,
                   fix_lon=False,fix_lat=False,fontsize=12):
    """
    Add Coastlines, grid, and set extent for geoaxes
    
    Parameters
    ----------
    ax : matplotlib geoaxes
        Axes to plot on 
    bbox : [LonW,LonE,LatS,LatN], optional
        Bounding box for plotting. The default is [-180,180,-90,90].
    proj : cartopy.crs, optional
        Projection. The default is None.
    blabels : ARRAY of BOOL [Left, Right, Upper, Lower] or dict
        Lat/Lon Labels. Default is [1,0,0,1]
    ignore_error : BOOL
        Set to True to ignore error associated with gridlabeling
    fill_color : matplotlib color string
        Add continents with a given fill
    c_zorder : layering order of the continents
    
    Returns
    -------
    ax : matplotlib geoaxes
        Axes with setup
    """
    
    if type(blabels) == dict: # Convert dict to array
        blnew = [0,0,0,0]
        if blabels['left'] == 1:
            blnew[0] = 1
        if blabels['right'] == 1:
            blnew[1] = 1
        if blabels['upper'] == 1:
            blnew[2] = 1
        if blabels['lower'] == 1:
            blnew[3] = 1
        blabels=blnew
    
    if proj is None:
        proj = ccrs.PlateCarree()
        
    if fill_color is not None: # Shade the land
        ax.add_feature(cfeature.LAND,facecolor=fill_color,zorder=c_zorder)
    #ax.add_feature(cfeature.COASTLINE,color=line_color,lw=0.75,zorder=0)
    ax.coastlines(color=line_color,lw=0.75)
    ax.set_extent(bbox,proj)
    
    gl = ax.gridlines(crs=proj, draw_labels=True,
                  linewidth=0.75, color=grid_color, alpha=0.5, linestyle="dotted",
                  )
    
    # Remove the degree symbol
    if ignore_error:
        #print("Removing Degree Symbol")
        gl.xformatter = LongitudeFormatter(zero_direction_label=False,degree_symbol='')
        gl.yformatter = LatitudeFormatter(degree_symbol='')
        #gl.yformatter = LatitudeFormatter(degree_symbol='')
        gl.rotate_labels = False
    
    if fix_lon is not False:
        gl.xlocator = mticker.FixedLocator(fix_lon)
    if fix_lat is not False:
        gl.ylocator = mticker.FixedLocator(fix_lat)
    
    gl.left_labels      = blabels[0]
    gl.right_labels     = blabels[1]
    gl.top_labels       = blabels[2]
    gl.bottom_labels    = blabels[3]
    
    # Set Fontsize
    gl.xlabel_style = {'size':fontsize}
    gl.ylabel_style = {'size':fontsize}
    return ax

def plot_mask(lon,lat,mask,reverse=False,color="k",marker="o",markersize=1.5,
              ax=None,proj=None,geoaxes=False):
    
    """
    Plot stippling based on a mask
    
    1) lon     [ARRAY] : Longitude values
    2) lat     [ARRAY] : Latitude values
    3) mask    [ARRAY] : (Lon,Lat) Mask (True = Where to plot Stipple)
    4) reverse [BOOL]  : Set to True to reverse the mask values
    5) color [STR] : matplotlib color
    6) marker [STR] : matplotlib markerstyle
    7) markersize [STR] : matplotlib markersize

    Solution from: https://matplotlib.org/stable/gallery/images_contours_and_fields/contour_corner_mask.html
    
    """
    if proj is None:
        if geoaxes:
            proj = ccrs.PlateCarree()
        else:
            proj = None
    # Get current axis
    if ax is None:
        ax = plt.gca()
        
    # Invert Mask
    if reverse:
        # Inversion doesnt work with NaNs...
        nlon,nlat = mask.shape
        newcopy   = np.zeros((nlon,nlat)) * np.nan
        newcopy[mask == True]  = False
        newcopy[mask == False] = True
        mask      = newcopy.copy()
    
    # Make meshgrid and plot masked array
    yy,xx    = np.meshgrid(lat,lon)
    if geoaxes:
        smap = ax.plot(np.ma.array(xx,mask=mask),yy,
                       c=color,marker=marker,markersize=markersize,ls="None",transform=proj)
    else:
        smap = ax.plot(np.ma.array(xx,mask=mask),yy,
                       c=color,marker=marker,markersize=markersize,ls="None")
    return smap 


#%% User Edits


# Calculation Options
leadlags          = np.arange(-3,4,1) # Lead-Lags for regression (in months)

# Regression Target Variable Infomration (Anomalized and Detrended variable in DataArray with dimensions [timename, latname, lonname])
expname           = "TCo319-DART-ssp585d-gibbs-charn"
time_name         = "time_counter"
lon_name          = 'lon'
lat_name          = 'lat'
timecrop          = None # Indicate Array with time crop [yearstart,yearend] if needed, ex: [1993,2004]
datpath           = "/home/niu4/gliu8/projects/scrap/regrid_1x1/global_anom_detrend1/" # Path to anomalized variable (ex. SST)
outpath           = "/home/niu4/gliu8/projects/scrap/" # Regression Output Path
vname             = "sst" # 
target_nc         = "%s%s_%s_regrid1x1.nc" % (datpath,expname,vname) # ex: <path_to_file>/TCo1279-DART-2060_sst_regrid1x1.nc

# ENSO Index Information (should have 'time') dimension
ensopath          = "/home/niu4/gliu8/projects/scrap/nino34/" # (!!) Set Path to ENSO Indices
ensoid_name       = "nino34" # Name of ENSO Index
standardize       = False # Set to True to return standardized ENSO Indices
enso_nc           = "%s%s_%s.nc" % (ensopath,expname,ensoid_name) # ex: <path_to_file>/TCo1279-DART-2060_nino34.nc



#%% Load ENSO ID

ds_enso          = xr.open_dataset(enso_nc).load()
# Chose whether or not to standardize the index
if standardize:
    ensoid      = ds_enso.sst 
else:
    ensoid      = ds_enso.sst * ds_enso['std'].data.item()

#%% Looping for each experiment

# Load DataArray
dsvar = xr.open_dataset(target_nc)[vname].load()

# Do Renaming
if time_name != 'time':
    dsvar = swap_rename(dsvar,time_name,'time')
if lon_name != 'lon':
    dsvar = swap_rename(dsvar,lon_name,'lon')
if lat_name != 'lat':
    dsvar = swap_rename(dsvar,lat_name,'lat')

# Crop time (mostly for control run, pre 1950)
if timecrop is not None:
    print("Cropping time for %s: %s to %s" % (expname,str(timecrop[0])+'-01-01',str(timecrop[1])+'-12-31'))
    dsvar = dsvar.sel(time=slice(str(timecrop[0])+'-01-01',str(timecrop[1])+'-12-31'))

# Check to make sure the time matches
dsvar,ensoid    = match_time_month(dsvar,ensoid)

# Get Dimension Lengths
dsvar           = dsvar.squeeze().transpose('lon','lat','time')
nlon,nlat,ntime = dsvar.shape

#%% Perform Lead-Lag Regression


# Do the Leads (variable leads)
leads       = np.abs(leadlags[leadlags <=0])
nleads      = len(leads)
beta_leads  = np.zeros((nlon,nlat,nleads)) * np.nan
sig_leads   = beta_leads.copy()
for ll in range(nleads):
    lag                = leads[ll] 
    ints               = ensoid.data[lag:]
    invar              = dsvar.data[:,:,:(ntime-lag)]
    rout               = regress_ttest(invar,ints,verbose=False)
    beta_leads[:,:,ll] = rout['regression_coeff']
    sig_leads[:,:,ll]  = rout['sigmask']
    
# Do the lags (timeseries leads)
lags        = leadlags[leadlags > 0]
nlags       = len(lags)
beta_lags   = np.zeros((nlon,nlat,nlags)) * np.nan
sig_lags    = beta_lags.copy()
for ll in range(nlags):
    lag   = lags[ll] 
    ints  = ensoid.data[:(ntime-lag)]
    invar = dsvar.data[:,:,lag:]
    rout  = regress_ttest(invar,ints,verbose=False)
    beta_lags[:,:,ll] = rout['regression_coeff']
    sig_lags[:,:,ll]  = rout['sigmask']

# Concatenate
betas = np.concatenate([beta_leads,beta_lags],axis=2)
sigs  = np.concatenate([sig_leads,sig_lags],axis=2)

# Replace into DataArray and Merge into Dataset
coords   = dict(lon=dsvar.lon,lat=dsvar.lat,lag=leadlags) 
da_betas = xr.DataArray(betas,coords=coords,dims=coords,name=dsvar.name) # Regression Slopes
da_sigs  = xr.DataArray(sigs,coords=coords,dims=coords,name="sig")       # Significance (from T-Test)
da_out   = [ds.transpose('lag','lat','lon') for ds in [da_betas,da_sigs]]
ds_out   = xr.merge(da_out)

#%% Save the Output

outname = "%sLagRegression_AllMonths_%s_%s_%s_standardize%i_lag%ito%i.nc" % (outpath,expname,vname,ensoid_name,standardize,leadlags[0],leadlags[-1])
edict   = make_encoding_dict(ds_out)
ds_out.to_netcdf(outname,encoding=edict)

#%% Part 2. Visualize Lag Regression

# Additional Inputs/Visualization choices
vunit   = "$\degree C$"
lag     = 0 # Lag to Visualize (note can turn this into loop)
projd   = ccrs.PlateCarree()
cmap    = 'RdBu_r'
vmax    = 1.5
cints   = np.arange(-1.5,1.8,3)

# Initialize Figure
fig,ax  = init_tp_map()
plotvar = ds_out[vname].sel(lag=lag)
sigin   = ds_out['sig']


    
# Plot the variable
pcm     = ax.pcolormesh(plotvar.lon,plotvar.lat,plotvar,
                        transform=projd,cmap=cmap,vmin=-vmax,vmax=vmax)

cl      = ax.contour(plotvar.lon,plotvar.lat,plotvar,
                     linewidths=0.55,levels=cints,
                        transform=projd,colors='k')
ax.clabel(cl)
    
# Plot Significance (stippling where points are not significant)
plotmask = sigin.sel(lag=lag)
lon      = plotmask.lon
lat      = plotmask.lat
if len(lon) < 500: # Adjust stippling based on length of longitude
    sigint = 1
elif len(lon) < 1000:
    sigint = 5
elif len(lon) < 4000:
    sigint = 20
else:
    sigint = 40

plot_mask(lon[::sigint],
        lat[::sigint],
        plotmask.T[::sigint,::sigint],
        reverse=False,ax=ax,proj=projd,geoaxes=True,markersize=0.65,color='gray')
    
cb      = fig.colorbar(pcm,ax=ax,fraction=0.01,pad=0.01)
cb.set_label("%s [%s per $1\sigma$ %s]" % (vname,vunit,ensoid_name))
ax.set_title("AWI-CM3 (%s) %s Regression, Lag %02i" % (expname,vname,lag))
    
plt.show()

