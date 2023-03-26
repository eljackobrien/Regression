# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:28:55 2020

@author: eljac
"""
from numpy import empty, hstack, repeat, argsort, sum, log10, mean, where, arange, inf, nan, array
from timeit import default_timer as timer
import matplotlib.pyplot as plt

# Keep track of the total function evaluations that are performed, less is better.
# -> For comparison between different guess values of optimisation algorithms.
num_evals = 0
def chisq(A, B):
    global num_evals
    num_evals += 1
    # Needed for low intensity RSM peaks where the noise causes many oscillations
    B[A==0] = 0
    return sum( (A - B)**2 )

algo = 'Nelder-Mead'

def get_psum(p):    # get psum which is needed to compute the reflection through the face
    global psum
    psum = sum(p, 0)

def reflect(func, x, z, I, p, p_all, vary_inds, vmin_arr, vmax_arr, chis, nmax, fac):
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
    get_psum(p)  # update psum with current values

    p_try = psum*fac1 - p[nmax]*fac2  # compute the reflected point
    # If parameters have exceeded the limits, then reset them to the limits
    min_inds, max_inds = p < vmin_arr, p > vmax_arr
    p[min_inds], p[max_inds] = vmin_arr[min_inds], vmax_arr[max_inds]

    p_all[vary_inds] = p_try
    chi_try = chisq(I, func(x, z, p_try))  # compute the chi value for this reflected point
    if chi_try < chis[nmax]:  # if this reflected point is better than the previous one
        chis[nmax] = chi_try
        p[nmax] = p_try
        get_psum(p)
    return chi_try


def NM_reg(func, x, z, I, p_all, s=None, fix=None, max_iter=1e4, ftol=1e-5, vmin0=None, vmax0=None,
           log=False, silence=False):
    '''func: function to fit to the data
    x:        x_data.
    z:        z_data.
    I:        I_data.
    p0:       guess_point.
    s:        scaling_factor.
    fix:      fixed_params.
    max_iter: max number of iteration over all params.
    ftol:     Fractional tolerance of function to break.
    vmin:     minimum values for the fit parameters.
    vmax:     maximum values for the fit parameters.
    log:      Option to record the variation of chi & ftol vs iteration number.\n
    Returns: optimal value of parameters p_opt to minimise function, chi_rtol'''
    p_all_copy = 1*p_all

    max_iter = int(max_iter)
    if max_iter==0: return p_all_copy, array([[chisq(I, func(x, z, p_all_copy)), nan]])

    t01 = timer()
    chi_rtol = empty((max_iter, 2))

    if type(s) == type(None): s = repeat(1/10, p_all_copy.shape)
    vmin = vmin0 if type(vmin0) != type(None) else repeat(-inf, len(p_all))
    vmax = vmax0 if type(vmax0) != type(None) else repeat( inf, len(p_all))
    vary_inds = where(fix == 0)[0] if type(fix) != type(None) else arange(len(p_all_copy))

    # Initialisation of simplex and chi values
    p0, s0 = p_all_copy[vary_inds], s[vary_inds]
    ndim, npts = p0.shape[0], p0.shape[0]+1
    vmin_arr, vmax_arr = repeat(vmin[None,:], npts, axis=0), repeat(vmax[None,:], npts, axis=0)

    p = repeat(p0, npts).reshape(ndim, npts).T
    for j in range(ndim):
        i = j + 1
        p[i, j] = p[i, j]*(1 + s0[j])  # Define the simplex, ndim+1 x ndim
    get_psum(p)
    chis = empty((npts))
    for k in range(npts):  # Calculate the initial chi values
        p_all_copy[vary_inds] = p[k]
        chis[k] = chisq(I, func(x, z, p_all_copy))

    if not silence: print('\nFitting via Nelder-Mead algorithm:')
    # Main optimisation loop
    for n_iter in range(max_iter):
        # Get highest, 2nd highest and lowest points in the simplex
        d = argsort(chis)
        nmax, nmax1, nmin = d[-1], d[-2], d[0]
        rtol = 2*abs((chis[nmax] - chis[nmin])/(abs(chis[nmax]) + abs(chis[nmin]) + ftol))  # Check relative tolerance
        if not n_iter%10 and not silence:
            print(f"\rIteration {n_iter}\t\t\u03C7\u00B2 = {chis[nmin]:.5g}\t\trtol = {rtol:.3e}    ", end="")
        chi_rtol[n_iter] = chis[nmin], rtol  # Save rtol & chi for plotting

        if rtol < ftol:  # Use rtol to break loop (Numerical Recipes recommended way)
            chi_rtol = chi_rtol[:n_iter]
            break  # Break loop if converged

        chi_try = reflect(func, x, z, I, p, p_all_copy, vary_inds, vmin_arr, vmax_arr, chis, nmax, -1.0)  # Get the reflected point
        if chi_try < chis[nmin]:    # If the new point (reflected from worst point) is now the best point, then extend the reflection further by a factor of 2
            chi_try = reflect(func, x, z, I, p, p_all_copy, vary_inds, vmin_arr, vmax_arr, chis, nmax, 2.0)
        elif chi_try > chis[nmax1]: # If new point is still the worst point (but better than previous worst point), half the distance the original reflection waa extended
            chi_save = chis[nmax]
            chi_try = reflect(func, x, z, I, p, p_all_copy, vary_inds, vmin_arr, vmax_arr, chis, nmax, 0.5)
            if chi_try > chi_save:  # If this half-length reflection is STILL shite, then give up and contract the simplex around the best point instead
                contract_inds = (chis != chis[nmin])
                p[contract_inds] = 0.5*(p[contract_inds] + p[nmin])

                # If parameters have exceeded the limits, then reset them to the limits
                min_inds, max_inds = p < vmin_arr, p > vmax_arr
                p[min_inds], p[max_inds] = vmin_arr[min_inds], vmax_arr[max_inds]

                for i in d[1:]:
                    p_all_copy[vary_inds] = p[i]
                    chis[i] = chisq(I, func(x, z, p_all_copy))
        get_psum(p)  # re-initialise psum with our new simplex

    p_all_copy[vary_inds] = p[nmin]

    if not silence:
        if n_iter == max_iter-1: print('\n\nMax iterations exceeded! :(')  # Check is loop above was broken -> converged
        else: print('\n\nOptimisation converged! :D')

        print('Total function evaluations: {}'.format(num_evals))
        t02 = timer()
        print('Fitting procedure took {:.2f} s = {:.2f} min'.format(t02-t01, (t02-t01)/60))
        final_chi2 = chisq( I, func(x, z, p_all_copy) )  # technically not chi2 definition, but avoids num/0
        R2 = 1 - sum( (I - func(x, z, p_all_copy))**2 ) / sum( (I - mean(I))**2 )
        print('Final \u03C7\u00B2 value\t=\t{:.4f}\nFinal R\u00B2 value\t=\t{:.4f}\n'.format(final_chi2, R2))

    if log:  # Plot the optimisation if log is true
        fig1, ax = plt.subplots(1, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
        ay = ax.twinx()
        ax.plot(log10(chi_rtol[:, 0]), c='b', label='log$_{10}$(chi)')
        ay.plot(log10(chi_rtol[:, 1]), c='r', label='log$_{10}$(rtol)')
        ay.plot(0, 0, c='b', label='log$_{10}$(chi)')
        ax.tick_params(axis='y', labelcolor='b'), ay.tick_params(axis='y', labelcolor='r')
        ax.set_xlabel('Iteration'), plt.title('log$_{10}$(Chi) & log$_{10}$(rtol) vs Iteration #')
        ax.grid(axis='both'), plt.legend(fontsize=14), plt.show()

    return p_all_copy, chi_rtol


def pr(temp, names, A):
    vals = 1*temp
    print('\nFit Values')
    if len(vals) == 17: vals[7], vals[15] = vals[7]/A, vals[15]/A
    elif len(vals) == 9: vals[7] = vals[7]/A
    temp = hstack((names.reshape(len(names), 1), vals.reshape(len(names), 1)))
    print(temp)




if __name__ in "__main__":
    import numpy as np
    from peak_functions import Gaussian_2D_general as G

    x, y = np.linspace(2, 6, 250), np.linspace(2, 6, 169)
    I = G(x, y, 15, 4.32, 3.96, 0.2, 0.532, 69) + \
        np.random.normal(scale=5.0, size=(len(y),len(x))) + 2.567

    def peak(x, y, vp): return vp[-1] + G(x, y, *vp[:-1])

    vg = np.array([1, 4.0, 4.0, 0.12, 0.08, 50, 2])
    vr, _ = NM_reg(peak, x, y, I, vg, s=None, fix=None, max_iter=1e4, ftol=1e-6, log=1, silence=0)

    fig, axes = plt.subplots(2,2, figsize=(14,14), tight_layout=True)
    axes.flatten()[0].contour(x, y, I)
    axes.flatten()[0].set_title("Data")
    axes.flatten()[1].contour(x, y, peak(x, y, vg))
    axes.flatten()[1].set_title("Guess Value")
    axes.flatten()[2].contour(x, y, peak(x, y, vr))
    axes.flatten()[2].set_title("Fit Value")
    axes.flatten()[3].set_axis_off()
    plt.show()

    print("Actual values = 15, 4.32, 3.96, 0.2, 0.532, 69, 2.567")
    print(f"Fit values = {np.around(vr, decimals=2)}")

