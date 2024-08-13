import os
import sys
import pandas as pd
import numpy as np

import astropy.units as u
from astropy.time import Time

os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".10"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
import optax
#print(jax.default_backend())


import jax.numpy as jnp

import jaxtromet
from jaxtromet.utils import hashdict

from jax import random,debug,jit
from jax.scipy.optimize import minimize
from jaxopt import ScipyMinimize,ScipyBoundedMinimize
from jaxopt import OptaxSolver
from jaxopt import ArmijoSGD
from jaxopt import LBFGSB

from jax import grad
from jax import device_get
from functools import partial

import scanninglaw.times as times
from scanninglaw.source import Source
from scipy.optimize import curve_fit
from sklearn.model_selection import ParameterGrid

jax.config.update("jax_enable_x64", True) # here changed to "jax.config" because deprecated
jax.config.update('jax_platform_name', 'cpu') # here changed to "jax.config" because deprecated

gpus = jax.devices('cpu')
print(gpus)


import time


############################################################
# chose parameters for the minimization
############################################################
l2_error_lim = 0.01 # limit on L2 opt. error
muwe_lim = 1.1 # limit on MUWE after minimization
choose_minimizer_global = 'l-bfgs-b' # choose minimization method: l-bfgs-b,BFGS,SLSQP,ArmijoSGD,optax_adam

lower_bounds = jnp.array([6.45,-47.6,-5,2014.6,0.,  -1,-1,-40,-40,0.5    ,0.1]) # lower limits for parameters
upper_bounds = jnp.array([6.55,-47.1, 5,2019.99,2.,   1, 1, 40, 40,1000.  ,10.]) # upper limits for parameters
#lower_bounds = jnp.array([6.45,-47.6,-5,2010.,0.,  -1,-1,-40,-40,0.5    ,0.1]) # lower limits for parameters
#upper_bounds = jnp.array([6.55,-47.1, 5,2024.,3.,   1, 1, 40, 40,1000.  ,12.]) # upper limits for parameters

bounds = (lower_bounds, upper_bounds)

#### Generate the parameter grid to cycle through different initial conditions ####
param_grid = {
    #'tE': np.linspace(jnp.min(4), jnp.max(4), 10),
    'u0': np.linspace(lower_bounds[2], upper_bounds[2], 10),
    'piEE': np.linspace(lower_bounds[5], upper_bounds[5], 5),
    'piEN': np.linspace(lower_bounds[6], upper_bounds[6], 5),
    'thetaE': np.linspace(lower_bounds[10], upper_bounds[10], 5), # optinally include this parameter
    't0': np.linspace(lower_bounds[3], upper_bounds[3], 5), # optinally include this parameter
    'tE': np.linspace(lower_bounds[4], upper_bounds[4], 5),# optinally include this parameter
}

grid = list(ParameterGrid(param_grid))

############################################################
# runtime options
############################################################
save_residual = False # True, False (calculate and save residuals, xobs, x_fit5)
use_my_5p = False # True, False (do not use 5-param fit values from Gaia data but from my own fit)
use_resid_parq = True # True, False (use residuals from .parquet files - not from my own fit)
change_guess_max = True # True, False (change initial guess of residuals to the maximum of mean residuals (not just the max of all residuals))

mock_obs_key, _ = random.split(random.PRNGKey(10), 2)

############################################################
# Function to update bounds on RA, DEC
############################################################
def update_bounds(ra_fit5, dec_fit5, lower_bounds, upper_bounds):
    lower_bounds_new = lower_bounds.at[0].set(ra_fit5 - 0.05)
    lower_bounds_new = lower_bounds_new.at[1].set(dec_fit5 - 0.05)
    upper_bounds_new = upper_bounds.at[0].set(ra_fit5 + 0.05)
    upper_bounds_new = upper_bounds_new.at[1].set(dec_fit5 + 0.05)
    return (lower_bounds_new, upper_bounds_new)

############################################################
# Define a standalone gradient function that doesn't capture external state (there was some memory leak with grad function before)
############################################################
def compute_gradient(params, tobs, xobs, phi_obs, err, bsfit, params_photo_fit, params_ext_fit):
    return grad(f_muwe_jaxopt)(params, tobs, xobs, phi_obs, err, bsfit, params_photo_fit, params_ext_fit)

############################################################
# JIT compile this function to avoid Python-level side effects
############################################################
jit_compute_gradient = jit(compute_gradient)

############################################################
# run minimizer with a method of you choice
############################################################
def optimize_(minimizer,lc,muwe_true,x0, bounds,t_obs, x_obs, phi_obs,
                    x_err, bs_fit, params_photo_fit,params_ext_fit, max_attempts=0,tol=1e-10,maxiter=2000):
    #LBFGS method with bound
    if minimizer=='l-bfgs-b':
        optimizer = ScipyBoundedMinimize(fun=f_muwe_jaxopt, method="l-bfgs-b", tol=tol,maxiter=maxiter)
        res = optimizer.run(init_params=x0, bounds=bounds,\
                            tobs=t_obs, xobs=x_obs, phi_obs=phi_obs,\
                            err=x_err, bsfit=bs_fit, params_photo_fit=params_photo_fit,\
                            params_ext_fit=params_ext_fit)

    #BFGS method (no bounds)
    elif minimizer=='BFGS':
        optimizer = ScipyMinimize(fun=f_muwe_jaxopt, method='BFGS')
        res = optimizer.run(init_params=x0,
                   tobs=t_obs, xobs=x_obs, phi_obs=phi_obs,
                   err=x_err, bsfit=bs_fit,params_photo_fit=params_photo_fit,
                   params_ext_fit=params_ext_fit)

    # least-square method
    elif minimizer=='SLSQP':
       optimizer = ScipyBoundedMinimize(fun=f_muwe_jaxopt, method='SLSQP')
       res = optimizer.run(init_params=x0,bounds=bounds,
                  tobs=t_obs, xobs=x_obs, phi_obs=phi_obs,
                  err=x_err, bsfit=bs_fit,params_photo_fit=params_photo_fit,
                  params_ext_fit=params_ext_fit)

    # SGD with Armijo line search
    elif minimizer=='ArmijoSGD':
        optimizer = ArmijoSGD(fun=f_muwe_jaxopt)
        res = optimizer.run(init_params=x0,
                   tobs=t_obs, xobs=x_obs, phi_obs=phi_obs,
                   err=x_err, bsfit=bs_fit,params_photo_fit=params_photo_fit,
                   params_ext_fit=params_ext_fit)

    # Stochastic optimization
    elif minimizer=='optax_adam':
        opt = optax.adam(learning_rate=0.01)
        optimizer = OptaxSolver(fun=f_muwe_jaxopt, opt=opt, maxiter=maxiter, tol=tol)
        res = optimizer.run(init_params=x0,
                   tobs=t_obs, xobs=x_obs, phi_obs=phi_obs,
                   err=x_err, bsfit=bs_fit,params_photo_fit=params_photo_fit,
                   params_ext_fit=params_ext_fit)

    #check muwe
    muwe_initial = f_muwe_jaxopt(res.params, t_obs, x_obs, phi_obs, x_err, bs_fit, params_photo_fit, params_ext_fit)
    muwe_min = muwe_initial

    # Checking the l2_optimality_error
    grad_val = jit_compute_gradient(res.params, t_obs, x_obs, phi_obs, x_err, bs_fit, params_photo_fit, params_ext_fit)
    l2_optimality_error_initial = jnp.linalg.norm(grad_val)
    L2_error=l2_optimality_error_initial
    print(f'Niter:{res.state.iter_num}, ID:{lc},MUWE_true:{muwe_true},MUWE:{muwe_initial},L2:{l2_optimality_error_initial}')

    # change initial conditions or perform test on events with a large L2 optimality error or large MUWE
    if l2_error_lim>0 and (l2_optimality_error_initial>l2_error_lim or muwe_min>muwe_lim):
        best_result = None
        best_L2_error = l2_optimality_error_initial  # try to reduce L2 opt. error

        for i, params in enumerate(grid):
            new_u0, new_piEE, new_piEN = params['u0'],  params['piEE'], params['piEN']

            # Set the new initial parameters
            if 'u0' in params:
                x0 = x0.at[2].set(new_u0)
            if 'piEE' in params:
                x0 = x0.at[5].set(new_piEE)
            if 'piEN' in params:
                x0 = x0.at[6].set(new_piEN)
            if 'thetaE' in params:
                new_thetaE=params['thetaE']
                x0 = x0.at[10].set(new_thetaE)
            if 't0' in params:
                new_t0=params['t0']
                x0 = x0.at[3].set(new_t0)
            if 'tE' in params:
                new_tE=params['tE']
                x0 = x0.at[4].set(new_tE)
            if 'thetaE' in params:
                print(f'Testing combination {i + 1}/{len(grid)}: u0={new_u0}, piEE={new_piEE}, piEN={new_piEN},thetaE={new_thetaE}, t0={new_t0}, tE={new_tE}')
            else:
                print(f'Testing combination {i + 1}/{len(grid)}: u0={new_u0}, piEE={new_piEE}, piEN={new_piEN}')

            # Run the optimizer with the new initial parameters
            if minimizer=='l-bfgs-b':
                res_new = optimizer.run(init_params=x0, bounds=bounds,
                                        tobs=t_obs, xobs=x_obs, phi_obs=phi_obs,
                                        err=x_err, bsfit=bs_fit, params_photo_fit=params_photo_fit,
                                        params_ext_fit=params_ext_fit)
            #BFGS method (no bounds)
            elif minimizer=='BFGS':
                res_new = optimizer.run(init_params=x0,
                           tobs=t_obs, xobs=x_obs, phi_obs=phi_obs,
                           err=x_err, bsfit=bs_fit,params_photo_fit=params_photo_fit,
                           params_ext_fit=params_ext_fit)
            # least-square method
            elif minimizer=='SLSQP':
               res_new = optimizer.run(init_params=x0,bounds=bounds,
                          tobs=t_obs, xobs=x_obs, phi_obs=phi_obs,
                          err=x_err, bsfit=bs_fit,params_photo_fit=params_photo_fit,
                          params_ext_fit=params_ext_fit)

            # SGD with Armijo line search.
            elif minimizer=='ArmijoSGD':
                res_new = optimizer.run(init_params=x0,
                           tobs=t_obs, xobs=x_obs, phi_obs=phi_obs,
                           err=x_err, bsfit=bs_fit,params_photo_fit=params_photo_fit,
                           params_ext_fit=params_ext_fit)

            # Stochastic optimization
            elif minimizer=='optax_adam':
                res_new = optimizer.run(init_params=x0,
                           tobs=t_obs, xobs=x_obs, phi_obs=phi_obs,
                           err=x_err, bsfit=bs_fit,params_photo_fit=params_photo_fit,
                           params_ext_fit=params_ext_fit)



            # Compute the L2 error of the gradient
            grad_val = jit_compute_gradient(res_new.params, t_obs, x_obs, phi_obs, x_err, bs_fit, params_photo_fit, params_ext_fit)
            L2_error = jnp.linalg.norm(grad_val)

            # Check if this run is better
            if L2_error < best_L2_error:
                best_L2_error = L2_error
                best_result = res_new

                #calculate MUWE
                muwe_min = f_muwe_jaxopt(res_new.params, t_obs, x_obs, phi_obs, x_err, bs_fit, params_photo_fit, params_ext_fit)

                # Exit the loop if a satisfactory solution is found
                if L2_error < l2_error_lim and muwe_min<muwe_lim:
                    print(f"Stopping early, satisfactory solution found with L2 error = {L2_error} and MUWE_min={muwe_min}")
                    break

        res = best_result
        muwe_min = f_muwe_jaxopt(res.params, t_obs, x_obs, phi_obs, x_err, bs_fit, params_photo_fit, params_ext_fit)
        print(f'Niter:{res.state.iter_num}, ID:{lc},MUWE_true:{muwe_true},MUWE:{muwe_min},L2:{L2_error}')

    return muwe_min,res,L2_error


############################################################
# necessary for Jaxtromet
############################################################
def convert_from_tuple(dic):
    phph = hashdict()
    for el in dic:
        if type(dic[el]) == tuple:
            #print(type(mw[el]))
            phph[el] = dic[el][0]
        else:
            phph[el] = dic[el]

    return phph


############################################################
# function that calculate MUWE
############################################################
def get_muwe_jaxopt(tobs: jnp.array,
             xobs: jnp.array,
             phi: jnp.array,
             err:  jnp.array,
             bsfit:  jnp.array,
             param_photo: dict,
             param_ext: dict
            ) -> float:
    # Assuming param_photo and param_ext are now regular dictionaries
    # Conversion from photometric to astrometric microlensing parameters
    t_fit = tobs
    param_astro = jaxtromet.define_lens(**param_photo)  # Ensure define_lens can handle unpacked arguments from a dictionary

    # Updating external parameters for binaries in the parameter dict()
    param_astro.update(param_ext)

    # Barycentric positions
    bs_fit = bsfit

    # Track generates per-transit coordinates and magnitudes using microlensing parameters
    ldracs, lddecs, mag_diff = jaxtromet.track(t_fit, bs_fit, dict(param_astro))

    # Lensed track without errors
    _, x_fit, phi_fit, rac_fit, dec_fit = jaxtromet.mock_obs(tobs, phi, ldracs, lddecs, 0, mock_obs_key, nmeasure=1)

    # Updated microlensing UWE
    muwe = jnp.sqrt(jnp.sum(((xobs - x_fit) / err) ** 2) / (jnp.size(xobs) - 11))

    return muwe


############################################################
# function that is minimized
############################################################
def f_muwe_jaxopt(x: jnp.array,
           tobs: jnp.array,
           xobs: jnp.array,
           phi_obs: jnp.array,
           err: jnp.array,
           bsfit: jnp.array,
           params_photo_fit: dict,
           params_ext_fit: dict) -> float:

    # Update photometric parameters using dictionary key access
    params_photo_fit['ra'] = x[0]
    params_photo_fit['dec'] = x[1]
    params_photo_fit['u0'] = x[2]
    params_photo_fit['t0'] = x[3]
    params_photo_fit['tE'] = x[4]
    params_photo_fit['piEN'] = x[5]
    params_photo_fit['piEE'] = x[6]
    params_photo_fit['pmrac_source'] = x[7]
    params_photo_fit['pmdec_source'] = x[8]
    params_photo_fit['d_source'] = x[9]
    params_photo_fit['thetaE'] = x[10]

    # Assuming get_muwe handles regular dictionaries for params_photo_fit and params_ext_fit
    muwe = get_muwe_jaxopt(tobs, xobs, phi_obs, err, bsfit, params_photo_fit, params_ext_fit)

    return muwe # Return both muwe and the parameters


############################################################
# generate x_fi5 - xobs for a 5 parameter (single source) model (needed for initial guess)
############################################################
def generate_xobs_single(id,tobs,phi,xobs,REFERENCE_EPOCH,g_mag,ra,dec,pmra,pmdec,parallax,u0,thetaE,t0,tE,piEE,piEN,param_ext):
     # convert t0 and tE
    t0 = Time(t0+2450000., format='jd').decimalyear
    tE = (tE*u.day).to(u.year).value

    mock_obs_key2, _ = random.split(random.PRNGKey(10), 2)

    # Conversion from photometric to astrometric microlensing parameters
    param_photo = hashdict(epoch = REFERENCE_EPOCH,fbl = 1)
    param_photo.ra = ra
    param_photo.dec = dec
    param_photo.u0 = u0
    param_photo.t0 = t0
    param_photo.tE = tE
    param_photo.piEN = piEN
    param_photo.piEE = piEE
    param_photo.pmrac_source = pmra
    param_photo.pmdec_source = pmdec
    param_photo.d_source = 1/abs(parallax) # 5-param fit can give negative parallax?
    param_photo.thetaE = thetaE

    print(f'id={id}, params from .parquet:',ra,dec,pmra,pmdec,parallax)
    params_jax = jaxtromet.define_lens(**convert_from_tuple(param_photo))

    t_fit = tobs

    # Updating external parameters for binaries in the parameter dict()
    params_jax.update(param_ext)

    # Barycentric positions
    bs_fit = jaxtromet.barycentricPosition(t_fit)

    # Track generates per-transit coordinates and magnitudes using microlensing parameters
    ldracs, lddecs, mag_diff = jaxtromet.track(t_fit, bs_fit, dict(params_jax))

    # Lensed track without errors
    _ ,x_fit ,phi_fit ,rac_fit ,dec_fit = jaxtromet.mock_obs(t_fit, phi, ldracs, lddecs, 0, mock_obs_key2, nmeasure = 1)

    # fit tracks and calculate UWE
    nmeasure=1
    magnitude_measurements = np.repeat(mag_diff, nmeasure)+g_mag
    ast_err = jaxtromet.sigma_ast(g_mag) # maybe better to use magnitude_measurements?
    results = jaxtromet.fit(t_fit, bs_fit,xobs, phi_fit,ast_err, ra, dec)
    print(f'id={id}, UWE from 5-parameter fit is:',results['uwe'])
    print(f'id={id}, params from fit:',results['ra_ref'],results['dec_ref'],results['pmrac'],results['pmdec'],results['parallax'])

    if jnp.isnan(results['uwe']):
        print('ID:',id,'UWE is nan. Params:',pmra,pmdec,parallax,', number of Gaia visits:',len(t_fit))
    return x_fit

############################################################
# estimate thetaE, t0, tE by fitting a Gaussian
############################################################
def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def estimate_lensing_gauss(id, t, phi_obs,x_obs, x_obs_est,resid, sigma_threshold, save_residual=False,num_bins=50,max_attempts=100):
    # Calculate residuals
    residuals = np.abs(x_obs - x_obs_est)

    # use residuals from .parquet files
    if use_resid_parq:
        residuals = np.abs(resid)
    max_all = np.max(residuals)

    # Histogram the time to create bins, then calculate averaged residuals per bin
    counts, bin_edges = np.histogram(t, bins=num_bins)
    #indices_non_zero=np.where(counts > 0.)[0] # exclude bins with 0 counts
    sums, _ = np.histogram(t, bins=bin_edges, weights=residuals)
    np.seterr(invalid='ignore') # Suppress division warnings
    averages = np.where(counts > 0, sums / counts, 0) # If a bin has no counts, the average is set to 0 to avoid division by zero.
    np.seterr(invalid='warn')# Restore numpy error settings
    #averages = sums[indices_non_zero] / counts[indices_non_zero]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # use maximum of mean residuals as an initial guess for the amplitude and time of maximum
    max_all = np.max(averages)
    t0_guess=bin_centers[np.where(averages==np.max(averages))[0][0]]

    # for the initial guess for sigma
    # Calculate mean and standard deviation of residuals
    #mean_residuals = np.mean(residuals)
    #std_residuals = np.std(residuals)
    mean_residuals = np.mean(averages)
    std_residuals = np.std(averages)

    # Find where residuals exceed the mean by the threshold of sigma
    #significant_indices = np.where(residuals > mean_residuals + sigma_threshold * std_residuals)[0]
    significant_indices = np.where(averages > mean_residuals + sigma_threshold * std_residuals)[0]

    if len(significant_indices) == 0:
        print("No significant microlensing event found")
        tE_guess = (bin_centers[-1] - bin_centers[0])
        start_index = 0
        end_index = -1
        t0_guess=2017.5
    else:
        start_index = significant_indices[0]
        end_index = significant_indices[-1]
        tE_guess = (bin_centers[end_index] - bin_centers[start_index])

    # Initial guesses for Gaussian fit
    initial_guess = [max_all,t0_guess, tE_guess*0.5]
    param_lim = ([0.8*max_all, lower_bounds[3], 0.2*tE_guess], [1.2*max_all+0.01, upper_bounds[3]+0.01, 2*tE_guess+0.01]) # duration of an event can be much longer than tE

    # Fit Gaussian to significant residuals
    success = False
    attempts = 0
    while not success and attempts < max_attempts:
        try:
            popt, pcov = curve_fit(gaussian, bin_centers, averages,p0=initial_guess, bounds=param_lim)
            perr = np.sqrt(np.diag(pcov))
            success = True
        except RuntimeError or ValueError as e:
            print(f"Attempt {attempts+1} failed: {e}")
            random_max = np.random.uniform(0.5 * max_all, 1.5 * max_all)
            random_t0 = np.random.uniform(lower_bounds[3], upper_bounds[3])
            random_sigma = np.random.uniform(lower_bounds[4], 10*upper_bounds[4])
            initial_guess = [random_max, random_t0, random_sigma]
            attempts += 1

    if not success:
        print("Failed to fit data after maximum attempts - setting estimates to initial guesses.")
        popt = initial_guess
        pcov = np.zeros((3, 3))
        perr = np.array([np.inf, np.inf, np.inf])

    thetaE, t0, tE = np.abs(popt[0]*3-2.25),popt[1],popt[2]
    error_thetaE, error_t0, error_sigma = perr
    if thetaE>10:
        print('thetaE to large. success, max_all, popt:',success, max_all, popt)

    # Optionally save residuals to file
    if save_residual:
        print('saving residuals to a .txt file')
        filename = f'Residual/estimate_params_{id}.txt'
        with open(filename, 'w') as f:
            header = "t,phi_obs,x_obs,x_fit5,residuals"
            f.write(header + '\n')
            for time, phi,obs, est, res in zip(t,phi_obs, x_obs, x_obs_est, residuals):
                f.write(f"{time},{phi},{obs},{est},{res}\n")

    return thetaE, Time(t0, format='decimalyear').jd -2450000, tE/365 # return in the correct format (t0 in JD, tE in days)

############################################################
# to read .parquet files and .csv files
############################################################
def get_data(fn,fdir,mic):

    dic = {}
    print("reading file "+fn)

    # generate ids and filenames
    tokens = fn.split('.parquet')
    id2 = tokens[1]
    p_fn = fdir+'/params.csv'+id2
    pfit_fn = fdir+'/params_fit.csv'+id2
    if mic:
        pm_fn = fdir+'/microlensing_params.csv'+id2
    if id2 == '':
        id2 = 'a'
    if id2 == '~':
        id2 = 'b'

    id2 = id2.replace('~','')
    id1 = tokens[0].split('_')[-1]

    if id2 not in dic:
        dic[id2] = {}

    dic[id2]['p_fn'] = p_fn
    dic[id2]['pfit_fn'] = pfit_fn
    if mic:
        dic[id2]['pm_fn'] = pm_fn # --> HERE!

    if 'lcs' not in dic[id2]:
        dic[id2]['lcs'] = {}

    if id1 not in dic[id2]['lcs']:
        dic[id2]['lcs'][id1] = {}

    dic[id2]['lcs'][id1]['fn'] = fn

    # read files
    for ex in dic:
        dic[ex]['df_p'] = pd.read_csv(dic[ex]['p_fn'])
        dic[ex]['df_pfit'] = pd.read_csv(dic[ex]['pfit_fn'])
        if mic:
            dic[ex]['df_pm'] = pd.read_csv(dic[ex]['pm_fn'])
    return dic

############################################################
# prepare output *_results.csv files
############################################################
def prepare_output_jaxopt(jres,smp_ind,lc_ind,m0,mt,l2_error,minimizer='lbfgsb',final_fun_val=[]): # final_fun_val for minimizer which don't have .fun_val
    # convert from Array to list of floats
    to_remove=['\n','[',']']
    res_str = str(jres.params)
    for char in to_remove:
        res_str = res_str.replace(char,'')

    res_str = res_str.strip()
    res_str = res_str.replace('         ',' ')
    res_str = res_str.replace('        ',' ')
    res_str = res_str.replace('       ',' ')
    res_str = res_str.replace('      ',' ')
    res_str = res_str.replace('     ',' ')
    res_str = res_str.replace('    ',' ')
    res_str = res_str.replace('   ',' ')
    res_str = res_str.replace('  ',' ')
    res_str = res_str.replace(' ',',')
    res_str

    if minimizer=='ArmijoSGD' or minimizer=='optax_adam' or minimizer=='lbfgsb_only':
        out_str = res_str+','+str(final_fun_val)
    else:
        out_str = res_str+','+str(jres.state.fun_val)
    out_lst = out_str.split(',')
    out_num = [float(x) for x in out_lst]

    out = []
    out.extend([smp_ind,lc_ind])
    out.extend(out_num)
    out.extend([m0,float(str(mt)),float(str(l2_error))])

    return out



def dump_to_disk(to_dump_lst,out_file,svmd,hdr):

    df_all = pd.DataFrame(to_dump_lst,columns=['sourceId','lc_id','ra', 'dec', 'u0', 't0', 'tE', 'piEN', 'piEE', 'pmrac_source', 'pmdec_source', 'd_source', 'thetaE','MUWE','MUWE_0','MUWE_true','L2_error'])

    df_all.to_csv(out_file,mode=svmd,header=hdr,index=False)



############################################################
# for testing
############################################################
def get_x_true(astro,photo):

    out = jnp.array([  astro['ra'].values[0],  # ra
                      astro['dec'].values[0],  # dec
                      photo['u_0'].values[0],  # u0
                      photo['t_0'].values[0],  # t0
                      photo['t_E'].values[0],  # tE
                    photo['pi_EN'].values[0],  # piEN
                    photo['pi_EE'].values[0],  # piEE
                     astro['pmra'].values[0],  # pmrac_source
                    astro['pmdec'].values[0],  # pmdec_source
               1/astro['parallax'].values[0],  # d_source
                  photo['theta_E'].values[0]   # thetaE
                   ])
    return out

############################################################
# main functions to process a single event
############################################################
def process_example(example,gpu_ind,mic):

    result = []

    for ex in example:
        true_a_par = example[ex]['df_p']
        fit5_a_par = example[ex]['df_pfit']
        if mic:
            true_p_par = example[ex]['df_pm']

        for lc in example[ex]['lcs']:

            print("processing: set="+str(ex)+", lc="+str(lc))

            microlensing_df = pd.read_parquet(example[ex]['lcs'][lc]['fn'],columns=['strip','sourceId','elapsedNanoSecs','w','wDiff','theta','wError','refEpoch','ra','dec','pmra','pmdec','varpi','GMag'])
            microlensing_df['strip'] = pd.Categorical(microlensing_df['strip'])
            microlensing_df['t_obs'] = (microlensing_df['elapsedNanoSecs'].values*u.nanosecond).to(u.year).value+2010.
            g_mag = microlensing_df['GMag'].iloc[0]
            residuals =jax.device_put(jnp.array(microlensing_df['wDiff']), jax.devices()[gpu_ind])
            source_ID = microlensing_df['sourceId'].iloc[0]

            # 5-parameter fit already in .parquet files
            ra_fit5 = microlensing_df['ra'].iloc[0]
            dec_fit5 = microlensing_df['dec'].iloc[0]
            pmra_fit5 = microlensing_df['pmra'].iloc[0]
            pmdec_fit5 = microlensing_df['pmdec'].iloc[0]
            parallax_fit5 = microlensing_df['varpi'].iloc[0]

            refEpoch = microlensing_df['refEpoch'].iloc[0]
            microlensing_df.drop(columns=['elapsedNanoSecs','refEpoch'],inplace=True)

            t_obs = jax.device_put(jnp.array(microlensing_df['t_obs']), jax.devices()[gpu_ind])
            x_obs = jax.device_put(jnp.array(microlensing_df['w']), jax.devices()[gpu_ind])
            # convert phi_obs to degrees if necessary (true Gaia data is in radians)
            gaia_true_data=False
            if jnp.max(jnp.array(microlensing_df['theta']))<jnp.pi:
                gaia_true_data = True
                phi_obs = jax.device_put(jnp.array(microlensing_df['theta'])*180./jnp.pi, jax.devices()[gpu_ind])
            else:
                phi_obs = jax.device_put(jnp.array(microlensing_df['theta']), jax.devices()[gpu_ind])
            x_err = jax.device_put(jnp.array(microlensing_df['wError']), jax.devices()[gpu_ind])

            bs_fit = jaxtromet.barycentricPosition(t_obs)

            # use my 5-parameter fit and not the one from Gaia (mainly for testing)
            if use_my_5p:
                results_5p = jaxtromet.fit(t_obs, bs_fit,x_obs, phi_obs,x_err, ra_fit5, dec_fit5)
                ra_fit5 = float(results_5p['ra_ref'])
                dec_fit5 = float(results_5p['dec_ref'])
                pmra_fit5 = float(results_5p['pmrac'])
                pmdec_fit5 = float(results_5p['pmdec'])
                parallax_fit5 = float(results_5p['parallax'])

            # Photometric parameters of microlensing event fit, blending parameter fbl is fixed to 1 fow now.
            # Take note that d_source_fit = 1/params_fit.parallax - i.e. it is a single parameter to fit.
            params_photo_fit = hashdict(
                epoch = refEpoch,
                fbl = 1
            )

            # External parameters for binaries (fixed values)
            params_ext_fit = hashdict(
                period = 1,   # year
                a = 0,        # AU
                e = 0.1,
                q = 0,
                l = 0,        # assumed < 1 (though may not matter)
                vtheta = jnp.pi/4,
                vphi = jnp.pi/4,
                vomega = 0,
                tperi = 0     # jyear
            )

            # for jaxopt
            params_photo_fit_jaxopt = {
                "epoch": refEpoch,
                 "fbl": 1
                 }

            # for jaxopt
            params_ext_fit_jaxopt = {
               "period": 1,
               "a": 0,
               "e": 0.1,
               "q": 0,
               "l": 0,
               "vtheta": jnp.pi / 4,
               "vphi": jnp.pi / 4,
               "vomega": 0,
               "tperi": 0
               }

            # estimate lens parameters from residuals
            x_obs_est = generate_xobs_single(lc,t_obs,phi_obs,x_obs,refEpoch,g_mag,ra_fit5,dec_fit5,pmra_fit5,pmdec_fit5,parallax_fit5,\
                                    u0=1000,thetaE=0.0001,t0=8000.,tE=300.,piEE=0.001,piEN=0.001,param_ext=params_ext_fit)

            # in some cases, the fit does not work
            if np.isnan(residuals[0]):
                thetaE_fit5,t0_fit5,tE_fit5 = 1, 8000,300
                print(f'id={lc}; residuals are nan - assuming arbitrary initial guess for thetaE, t0, tE')
            else:
                thetaE_fit5,t0_fit5,tE_fit5=estimate_lensing_gauss(lc,t_obs,phi_obs, x_obs, x_obs_est,residuals, sigma_threshold=1,save_residual=save_residual)
            thetaE_true,t0_true,tE_true=thetaE_fit5,t0_fit5,tE_fit5
            print(f'id={lc}; estimated thetaE,t0,tE:',thetaE_true,Time(t0_true+2450000., format='jd').decimalyear,(tE_true*u.day).to(u.year).value)

            # initial values
            x0 = jax.device_put(jnp.array([   ra_fit5,  # ra
                                             dec_fit5,  # dec
                                                  0.001,  # u0 ;u0_true 0.001
                                              t0_true,  # t0
                                              tE_true,  # tE
                                            0.001,  # piEN;piEN_true 0.001
                                            0.001,  # piEE;piEE_true 0.001
                                            pmra_fit5,  # pmrac_source
                                           pmdec_fit5,  # pmdec_source
                                    1.0/parallax_fit5,  # d_source
                                          thetaE_true   # thetaE
                                           ]),jax.devices()[gpu_ind])

            # convert t0 and tE
            x0 = x0.at[3].set(Time(x0[3]+2450000., format='jd').decimalyear)
            x0 = x0.at[4].set((x0[4]*u.day).to(u.year).value)

            # get initial muwe jaxopt
            if gaia_true_data:
                muwe_true = 1 # set an arbitrary value
            else:
                x_true = jax.device_put(get_x_true(true_a_par.iloc[[int(lc)-1]],true_p_par.iloc[[int(lc)-1]]),jax.devices()[gpu_ind]) # --> HERE!
                # convert t0 and tE
                x_true = x_true.at[3].set(Time(x_true[3]+2450000., format='jd').decimalyear)
                x_true = x_true.at[4].set((x_true[4]*u.day).to(u.year).value)
                muwe_true = f_muwe_jaxopt(x_true,t_obs,x_obs,phi_obs,x_err,bs_fit,params_photo_fit,params_ext_fit)
            muwe_0 = f_muwe_jaxopt(x0,t_obs, x_obs, phi_obs, x_err, bs_fit, params_photo_fit_jaxopt, params_ext_fit_jaxopt)

            # make upper and lower bounds of ra, dec a bit more flexible - this is pretty much fixed
            bounds_new = update_bounds(ra_fit5, dec_fit5, lower_bounds, upper_bounds)

            #minimization with jaxopt
            choose_minimizer_local=choose_minimizer_global
            muwe_min,res,l2_optimality_error= optimize_(choose_minimizer_local,lc,muwe_true,x0, bounds_new,t_obs, x_obs, phi_obs,
                    x_err, bs_fit, params_photo_fit_jaxopt,params_ext_fit_jaxopt, max_attempts=10)

            # jaxopt minimizer
            if choose_minimizer_local=='ArmijoSGD' or choose_minimizer_local=='optax_adam' or choose_minimizer_local=='lbfgsb_only':
                final_fun_value= f_muwe_jaxopt(res.params, t_obs, x_obs, phi_obs, x_err, bs_fit, params_photo_fit_jaxopt, params_ext_fit_jaxopt)
                out = prepare_output_jaxopt(res,source_ID,lc,muwe_0,muwe_true,l2_optimality_error,choose_minimizer_local,final_fun_value) # --> HERE!
            else:
                out = prepare_output_jaxopt(res,source_ID,lc,muwe_0,muwe_true,l2_optimality_error) # --> HERE!
            result.append(out)

    return result


####################################333



if __name__ == "__main__":


    #sys.exit()
    input_file = sys.argv[1]
    input_dir = sys.argv[2]
    output_file = sys.argv[3]
    write_mode = sys.argv[4]
    gpu_id = int(sys.argv[5])
    micro = sys.argv[6]
    if micro=="True":
        print("Working with microlensing!")
        micro=True
    else:
        print("NOT working with microlensing!")
        micro=False

    print("input_file = " + input_file)
    print("input_dir = " + input_dir)
    print("output_file = " + output_file)
    print("write_mode = " + write_mode)
    print("gpu_id = " + str(gpu_id))
    print("micro = " + str(micro))
    #sys.exit()

    start_time1 = time.perf_counter()
    example = get_data(input_file,input_dir,micro)
    finish_time1 = time.perf_counter()

    start_time2 = time.perf_counter()
    results = process_example(example,gpu_id,micro)
    finish_time2 = time.perf_counter()

    start_time3 = time.perf_counter()
    if write_mode == 'w':
        header = True
    else:
        header = False
    # save results
    dump_to_disk(results,output_file,write_mode,header)
    #dump_to_disk_muwe_true(results,output_file,write_mode,header)

    finish_time3 = time.perf_counter()


    print("reading data from disk: {} seconds".format(finish_time1-start_time1))
    print("processing data: {} seconds".format(finish_time2-start_time2))
    print("saving data to disk: {} seconds".format(finish_time3-start_time3))
    print("total time: {} seconds".format(finish_time3-start_time1))
    #print()
    #print()
