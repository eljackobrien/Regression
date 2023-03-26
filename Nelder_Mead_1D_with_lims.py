# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:28:55 2020

@author: eljac
"""
from numpy import empty, repeat, transpose, argsort, sum, inf, log10, where, arange, mean
from timeit import default_timer as timer
import matplotlib.pyplot as plt

# Simple Chi calculator
def chisq(A, B):
    global num_evals
    num_evals += 1
    return sum( (A - B)**2 )

algo = 'Nelder-Mead'


def get_psum(p):    # get psum which is needed to compute the reflection through the face
    global psum
    psum = sum(p, 0)


def reflect(func, x, I, p, p_all, vary_inds, vmin_arr, vmax_arr, chis, nmax, fac):
    ''' Reflect highest point of simplex through opposite face, extend by factor 'fac'.
    p:     ndim+1 vectors of length ndim each (i.e. 6 points for 5 variables).
    chis:  chi value (function value) for each of the initial ndim points.
    nmax:  position of the maximum point in the simplex.
    fac:   factor to extend the point through the face (-1 for reflection).\n
    Returns: new simplex with highest point reflected through face'''
    ndim = p.shape[1]

    p_try = empty((ndim))
    fac1 =  (1 - fac)/ndim
    fac2 = fac1 - fac
    get_psum(p)    # update psum with current values
    p_try = psum*fac1 - p[nmax]*fac2  # compute the reflected point
    # If parameters have exceeded the limits, then reset them to the limits
    min_inds, max_inds = p < vmin_arr, p > vmax_arr
    p[min_inds], p[max_inds] = vmin_arr[min_inds], vmax_arr[max_inds]

    p_all[vary_inds] = p_try
    chi_try = chisq(I, func(x, p_all))              # compute the chi value for this reflected point
    if chi_try < chis[nmax]:    # if this reflected point is better than the previous one
        chis[nmax] = chi_try
        p[nmax] = p_try
        get_psum(p)
    return chi_try


def NM_reg(func, x, I, p0, s=None, fix=None, max_iter=1e4, ftol=1e-5,
           vmin0=None, vmax0=None, log=False, silence=False):
    '''func: function to fit to the data
    x:        x_data.
    I:        I_data.
    p_inp:    Guess_point, including those which will not be varied.
    s:        Initial step size for regression.
    fix:      Fixed parameter indices.
    max_iter: Max number of iteration over all params.
    ftol:     Fractional tolerance of function to break.
    vmin0:    Minimum value for each parameter
    vmax0:    Minimum value for each parameter
    log:      Option to plot the variation of chi & ftol vs iteration number.
    silence:  Silences the printing of progress to the console.\n
    Returns:  [Optimal value of parameters p_opt to minimise function, progression of rtol]'''
    # Don't want p0 to be altered in-place, so we use a dummy variable and multiply
    p_all = 1*p0

    global num_evals
    num_evals = 0
    max_iter = int(max_iter)
    if max_iter < 1:
        print("max_iter < 1 specified, no fitting performed")
        return p_all
    t01 = timer()
    chi_rtol = empty((max_iter, 2))

    # Set values for any undefined control arrays
    if type(s) == type(None): s = repeat(1/10, len(p_all))
    vary_inds = where(fix == 0)[0] if type(fix) != type(None) else arange(len(p_all))
    vmin = vmin0 if type(vmin0) != type(None) else repeat(-inf, len(p_all))
    vmax = vmax0 if type(vmax0) != type(None) else repeat( inf, len(p_all))
    if not len(vary_inds): return p_all

    # Initialisation of simplex and chi values
    p0, s0 = p_all[vary_inds], s[vary_inds]
    ndim, npts = p0.shape[0], p0.shape[0]+1
    vmin_arr, vmax_arr = repeat(vmin[vary_inds][None,:], npts, axis=0), repeat(vmax[vary_inds][None,:], npts, axis=0)

    p = transpose(repeat(p0, npts).reshape(ndim, npts))
    for j in range(ndim):
        i = j + 1
        p[i, j] = p[i, j]*(1 + s0[j])   # Define the simplex, ndim+1 x ndim
    get_psum(p)
    chis = empty((npts))
    for k in range(npts)[::-1]:         # Calculate the initial chi values
        p_all[vary_inds] = p[k]
        chis[k] = chisq(I, func(x, p_all))

    if not silence: print('\nFitting via Nelder-Mead algorithm:')
    # Main optimiastion loop
    for n_iter in range(max_iter):
        # Get highest, 2nd highest and lowest points in the simplex
        d = argsort(chis)

        nmax, nmax1, nmin = d[-1], d[-2], d[0]
        rtol = 2*abs((chis[nmax] - chis[nmin])/(abs(chis[nmax]) + abs(chis[nmin]) + ftol))  # Check relative tolerance
        if not n_iter%10 and not silence: print('\rIteration {}\t\t\tChi= {:.4g}\t\trtol = {:.3g}'.format(n_iter, chis[nmin], rtol), end="")  # Print some messages, every 10 iterations
        chi_rtol[n_iter] = chis[nmin], rtol  # Save rtol & chi for plotting

        if rtol < ftol:  # Use rtol to break loop (Numerical Recipes recommended way)
            chi_rtol = chi_rtol[:n_iter]
            break  # Break loop if converged

        # p is the simplex, made up of only the varying points, p_all is a 1D matrix of all the necessary points to calculate chi
        chi_try = reflect(func, x, I, p, p_all, vary_inds, vmin_arr, vmax_arr, chis, nmax, -1.0)  # Get the reflected point
        if chi_try < chis[nmin]:    # If the new point (reflected from worst point) is now the best point, then extend the reflection further by a factor of 2
            chi_try = reflect(func, x, I, p, p_all, vary_inds, vmin_arr, vmax_arr, chis, nmax, 2.0)
        elif chi_try > chis[nmax1]: # If new point has improved but is still the worst point half the distance the original reflection waa extended
            chi_save = chis[nmax]
            chi_try = reflect(func, x, I, p, p_all, vary_inds, vmin_arr, vmax_arr, chis, nmax, 0.5)
            if chi_try > chi_save:
                # If this half-length reflection is STILL shite, then give up and contract the simplex around the best point instead
                contract_inds = (chis != chis[nmin])
                p[contract_inds] = 0.5*(p[contract_inds] + p[nmin])
                # If parameters have exceeded the limits, then reset them to the limits
                min_inds, max_inds = p < vmin_arr, p > vmax_arr
                p[min_inds], p[max_inds] = vmin_arr[min_inds], vmax_arr[max_inds]

                for i in d[1:]:
                    p_all[vary_inds] = p[i]
                    chis[i] = chisq(I, func(x, p_all))
        get_psum(p)  # re-initialise psum with our new simplex

    p_all[vary_inds] = p[nmin]

    if not silence:
        if n_iter == max_iter-1: print('\n\nMax iterations exceeded! :(')  # Check is loop above was broken -> converged
        else: print('\n\nOptimisation converged! :D')

        print('Total function evaluations: {}'.format(num_evals))
        t02 = timer()
        print('Fitting procedure took {:.2f} s = {:.2f} min'.format(t02-t01, (t02-t01)/60))
        final_chi2 = chisq( I, func(x, p_all) )  # technically not chi2 definition, but avoids num/0
        R2 = 1 - sum( (I - func(x, p_all))**2 ) / sum( (I - mean(I))**2 )
        print('Final \u03C7\u00B2 value\t=\t{:.4f}\nFinal R\u00B2 value\t=\t{:.4f}\n'.format(final_chi2, R2))

    if log:  # Plot the optimisation progress if log is true
        fig1, ax = plt.subplots(1, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
        ay = ax.twinx()
        ax.plot(log10(chi_rtol[:, 0]), c='b', label='log$_{10}$(chi)')
        ay.plot(log10(chi_rtol[:, 1]), c='r', label='log$_{10}$(rtol)')
        ay.plot(0, 0, c='b', label='chi')
        ax.tick_params(axis='y', labelcolor='b'), ay.tick_params(axis='y', labelcolor='r')
        ax.set_xlabel('Iteration'), plt.title('log$_{10}$(Chi) & log$_{10}$(rtol) vs Iteration #')
        ax.grid(axis='both'), plt.legend(), plt.show()

    return p_all, chi_rtol





#%% Minimise version of the functions (as opposed to fitting to data)

#  In order to find the minimum of a function with no x values, just f(a, b, c)
#  just define a dummy function which takes an x array but does nothing
#  def dummy(x, a,b,c): return func(a,b,c)
#
#  Then for the y/z values, just enter a large negative value and therefore the optimisation will return
#  the minimum of the function, since that is closest to this large negative value.



