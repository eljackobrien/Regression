# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 21:13:39 2022

@author: eljac
"""
import numpy as np


def red_chi_sq(x_data, y_data, func, params):
    DOF = len(x_data) - len(params)
    y_fit = func(x_data, params)
    resid = y_fit - y_data
    chi_sq = np.sum(resid**2)
    global num_evals
    num_evals += 1
    return chi_sq / DOF


def particle_swarm_optimise(x_data, y_data, func, params_min, params_max, max_evals,
                            num_particles=50, inertia=0.5, cognitive=1, social=1,
                            limits='soft', return_iters=False):
    """
    Find the optimal function parameters such that func most closely approximates y_data.

    Parameters
    ----------
    x_data : np.ndarray
        Independent variable to be passed into function.
    y_data : np.ndarray
        Dependent variable to be approximated by function.
    func : function
        The function whose return values we are approximating y_data with.
    params_min : np.ndarray
        Minimum expected values for the function parameters.
    params_max : np.ndarray
        Maximum expected values for the function parameters.
    max_evals : int
        Maximum number of function evaluations before returning.
    num_particles : int, optional
        Number of (uniform randomly distributed in parameters space) particles to use. The default is 50.
    inertia : float, optional
        Influence of particles previous velocity on new velocity [0,0.95]. The default is 0.5.
    cognitive : float, optional
        Influence of distance to best visited location per particle on new velocity [1,3]. The default is 1.
    social : float, optional
        Influence of distance to global best location of all particles on new velocity [1,3]. The default is 1.
    limits : str, optional
        ['hard', 'soft', 'none'] How to deal with particles going outside the parameter limits. The default is 'soft'.
    return_iters : bool, optional
        Whether or not to return the particle positions/velocities at each iteration. The default is False.

    Returns
    -------
    global_min : np.ndarray
        parameters which give the minimum reduced chi**2 for func(x_data, params) - y_data.
        if return_iters is true, a tuple of the above and a list of position and velocity for each iter.
    """

    # Convert lists/floats to arrays/ints
    if type(x_data) != np.ndarray: x_data = np.array(x_data)
    if type(y_data) != np.ndarray: y_data = np.array(y_data)
    if type(params_min) != np.ndarray: params_min = np.array(params_min)
    if type(params_max) != np.ndarray: params_max = np.array(params_max)
    max_evals = int(max_evals)
    global num_evals
    num_evals = 0

    # N number of particles and number of parameters
    N_PNTS = int(num_particles)
    N_PARAMS = len(params_min)
    # array versions for params_min and max for easy vector comparisons with pos
    par_min_arr, par_max_arr = np.repeat(params_min[None,:], N_PNTS, axis=0), np.repeat(params_max[None,:], N_PNTS, axis=0)

    # cognitive and social coefficient of swarm, typically 1 < c < 3
    # Dictates relative weighting of best_visited and global_best position for each particle
    C_COG, C_SOC = cognitive, social

    # initialise particle positions uniform-randomly within the given limits, store best-visited
    pos = np.random.uniform(params_min, params_max, (N_PNTS, N_PARAMS))
    best_pos = 1*pos
    # assuming range = (high_lim - low_lim), initialise velocity between -range and range
    params_range = abs(params_max - params_min)
    vel = np.random.uniform(-params_range, params_range, (N_PNTS, N_PARAMS))

    # Evaluate the function at each particle position, store current and previous values
    best_chisq = np.empty(N_PNTS)
    for i in range(N_PNTS): best_chisq[i] = red_chi_sq(x_data, y_data, func, pos[i])
    chisq_crnt = 1*best_chisq
    global_min = pos[np.argmin(chisq_crnt)]

    if return_iters: p_iter, v_iter = [pos], [vel]
    # Loop let the particles move around through the space until maximum function evaluations
    while num_evals <= (max_evals-N_PNTS):
        # Particles velocity changes based on:
        #     inertia: weighting given to previous velocity (must be <1 to prevent divergence)
        #     cognitive coefficient: weighting given to particles best visited position
        #     social coefficient: weighting given to global best position
        w, [rnd_cog, rnd_soc] = inertia, np.random.uniform(size=(2,N_PNTS))
        cognitive_vel = C_COG * rnd_cog[:,None] * (best_pos - pos)
        social_vel = C_SOC * rnd_soc[:,None] * (global_min - pos)
        vel = w*vel + cognitive_vel + social_vel

        # update the particles positions and evaluate function for each
        pos = pos + vel
        # Set params which have strayed outside limits, equal to the limits
        if limits=='hard':
            pos[pos < params_min] = par_min_arr[pos < params_min]
            pos[pos > params_max] = par_max_arr[pos > params_max]
        # Reverse velocity of parameters which have strayed outside limits
        elif limits=='soft':
            vel[pos < params_min] = -vel[pos < params_min]
            vel[pos > params_max] = -vel[pos > params_max]

        for i in range(N_PNTS): chisq_crnt[i] = red_chi_sq(x_data, y_data, func, pos[i])
        chisq_crnt[np.isnan(chisq_crnt)] = np.inf # dirty way to take care of NANs for now
        # For which particles is the new position the best-visited
        inds = chisq_crnt < best_chisq
        # Store new best-visited positions and chisq
        best_pos[inds], best_chisq[inds] = pos[inds], chisq_crnt[inds]
        # Update global minimum
        global_min = best_pos[np.argmin(best_chisq)]

        if return_iters: p_iter.append(pos), v_iter.append(vel)

    if return_iters: return global_min, [p_iter, v_iter]
    return global_min




#%% Test the performance

if __name__ in "__main__":
    import matplotlib.pyplot as plt

    #%% Textbook examplle - simple 2D minimisation with multiple local minima
    def f(x,y): return (x-3.14)**2 + (y-2.72)**2 + np.sin(3*x+1.41) + np.sin(4*y-1.73)
    def test_func(x, params): return f(*params)

    # Calculate for the contour map
    xr, yr = np.linspace(-1, 6, 100), np.linspace(-1, 5, 100)
    xrg, yrg = np.meshgrid(xr, yr); pars = [xr, yrg]#np.vstack((xr, yr))
    z = test_func(xr, pars)

    x, y = np.zeros(10), np.zeros(10)-10
    params_min, params_max = [-1, -1], [6, 5]
    params_opt, [p_iter,v_iter] = particle_swarm_optimise(x, y, test_func, params_min, params_max, 1e3,
                                         num_particles=30, inertia=0.40, cognitive=1.25, social=1.0,
                                         limits='soft', return_iters=True)

    # Use the interactive qt backend plotting to have updating plot
    try:
        import IPython
        shell = IPython.get_ipython()
        shell.enable_matplotlib(gui='qt')
        fig,ax = plt.subplots(1,1, figsize=(8,8))
        plt.title("Surface map with particles & velocities", fontsize=16)
        plt.xlim(params_min[0], params_max[0]), plt.ylim(params_min[1], params_max[1])
        plt.contourf(xr, yr, z, levels=20, alpha=0.7, cmap="Oranges")
        plt.contour(xr, yr, z, levels=20, alpha=0.5, cmap="gray")
        plt.scatter(3.182, 3.131, s=10**2, marker='x',lw=3.0, c='g', label='actual min')

        pos, = ax.plot(p_iter[0][:,0], p_iter[0][:,1], ls='', marker='o', c='royalblue')
        vec = ax.quiver(p_iter[0][:,0], p_iter[0][:,1], v_iter[0][:,0], v_iter[0][:,1], scale_units='xy', scale=2, width=0.003)
        for p_iters,v_iters in zip(p_iter,v_iter):
            # Update the data stored in the plot objects
            pos.set_data(p_iters[:,0], p_iters[:,1])
            vec.set_offsets(p_iters)
            vec.set_UVC(v_iters[:,0], v_iters[:,1])
            # Update the drawn plot
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.2)

        shell.enable_matplotlib(gui='inline')

    # Use the default plotting and make separate plots (inline in spyder)
    except:
        for p_iters,v_iters in zip(p_iter,v_iter):
            plt.figure(figsize=(8,8)), plt.title("Surface map with particles & velocities", fontsize=16)
            plt.xlim(params_min[0], params_max[0]), plt.ylim(params_min[1], params_max[1])
            plt.contourf(xr, yr, z, levels=20, alpha=0.7, cmap="Oranges")
            plt.contour(xr, yr, z, levels=20, alpha=0.5, cmap="gray")

            plt.scatter(p_iters[:,0], p_iters[:,1], c='royalblue')
            plt.quiver(p_iters[:,0], p_iters[:,1], v_iters[:,0], v_iters[:,1], scale_units='xy', scale=2, width=0.003)
            plt.scatter(3.182, 3.131, s=10**2, marker='x',lw=3.0, c='g', label='actual min')

            plt.show()


    """ Save an animation of the particle swarm

    from matplotlib.animation import FuncAnimation

    fig,ax = plt.subplots(1,1, figsize=(8,8))
    plt.title("Surface map with particles & velocities", fontsize=16)
    plt.xlim(params_min[0], params_max[0]), plt.ylim(params_min[1], params_max[1])
    plt.contourf(xr, yr, z, levels=20, alpha=0.7, cmap="Oranges")
    plt.contour(xr, yr, z, levels=20, alpha=0.5, cmap="gray")
    plt.scatter(3.182, 3.131, s=10**2, marker='x',lw=3.0, c='g', label='actual min')

    pos, = ax.plot(p_iter[0][:,0], p_iter[0][:,1], ls='', marker='o', c='royalblue')
    vec = ax.quiver(p_iter[0][:,0], p_iter[0][:,1], v_iter[0][:,0], v_iter[0][:,1], color='navy',
                    scale_units='xy', scale=1.5, width=0.0035)

    def anim_func(i):
        pos.set_data(p_iter[i][:,0], p_iter[i][:,1])
        vec.set_offsets(p_iter[i])
        vec.set_UVC(v_iter[i][:,0], v_iter[i][:,1])

    anim = FuncAnimation(fig, anim_func, frames=len(p_iter), interval=200)
    anim.save('particle_swarm_optimisation_example.mp4', writer='ffmpeg', fps=6)

    """

    #%% Test voigt fitting function - 4 parameters (compare with NM)
    from peak_functions import pV as psuedo_voigt
    import Nelder_Mead_1D_with_lims as NM
    # Define function and data for testing purposes, fitting MRG_206 data with a psuedo-voigt
    def test_func(x, params):
        slope, inter, x0, g, l, A = params
        return inter + slope*x + psuedo_voigt(x, x0, g, l, A) # params: [slope, inter, x0, g, l, A]
    test_path = "C:/Users/eljac/Documents/College_Stuff/1_Ph.D._Stuff/XRay_Analysis/data/MRG/GA200521E_RSM/206qz_peak.xy"
    test_data = np.genfromtxt(test_path)
    test_data = test_data[test_data[:,1] > 0]
    x, y = test_data[:,0], test_data[:,1]

    mx = np.argmax(y)
    #params_name =          slope,   inter,  x0,         sigma,   gamma,   amp
    params_min =  np.array([-3,     -3,      x[mx]-0.1,  0.001,   0.001,   20])
    params_max =  np.array([3,       5,      x[mx]+0.1,  0.08,    0.08,    70])

    params_opt = particle_swarm_optimise(x, y, test_func, params_min, params_max, 2e4,
                                         num_particles=100, inertia=0.75, cognitive=2, limits='hard')
    # Nelder-mead algorithm is FAR faster for something like this but sure look
    p_opt2, _ = NM.NM_reg(test_func, x, y, (params_max+params_min)/2)

    chisq = red_chi_sq(x, y, test_func, params_opt)
    print(f"\nnum_evals = {num_evals}\nBest_chisq = {chisq:.3g}")

    # Plot the optimised parameters relative to the initial limits
    plt.figure(figsize=(8,5)), plt.title("Opt params with limits")
    norm = (params_opt - np.array(params_min)) / (np.array(params_max) - np.array(params_min))
    plt.plot(norm, marker='o', ms=6, linestyle='', c='b', label='Opt', zorder=5)
    plt.axhline(0, lw=1.5, c='r'), plt.axhline(1, lw=1.5, c='r'), plt.grid(lw=1.0, zorder=-5)
    plt.legend(fontsize=16), plt.show()

    # Plot the data and fit
    y_fit, y_fit_NM = test_func(x, params_opt), test_func(x, p_opt2)
    plt.figure(figsize=(10,6.5), constrained_layout=True)
    plt.scatter(x, y, facecolors='none', edgecolors='red', s=6**2, lw=1.0, label='Data')
    plt.plot(x, y_fit, lw=2.0, c='limegreen', label='Fit-PS')
    plt.plot(x, y_fit_NM, lw=2.5, ls=(0,(5,5)), c='royalblue', label='Fit-NM')
    plt.legend(fontsize=20), plt.show()

