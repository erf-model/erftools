from importlib import resources
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_1d(temp_3d, pressure_3d, theta_3d, qv_3d, qsat_3d, z_grid,
            k_to_delete=[], figdir='./Images'):
    """Plot vertical profiles of atmospheric quantities"""
    os.makedirs(figdir, exist_ok=True)

    nz = z_grid.shape[2]

    plt.figure(1)
    for k in np.arange(nz-1, -1, -1):
        if k in k_to_delete:
            continue
        plt.plot(np.mean(temp_3d[:,:,k]), np.mean(z_grid[:,:,k]), 'xk', label="mean" if k == 0 else "")
        plt.plot(np.max( temp_3d[:,:,k]), np.mean(z_grid[:,:,k]), 'or', label="max"  if k == 0 else "")
        plt.plot(np.min( temp_3d[:,:,k]), np.mean(z_grid[:,:,k]), '^b', label="min"  if k == 0 else "")
    plt.ylim([0, 20000])
    plt.xlabel('T (K)',fontsize=15)
    plt.ylabel(r'$z$ (m)',fontsize=15)

    with resources.open_text('erftools.data.typical_atmosphere', 'temp_vs_z_actual.txt') as f:
        data = np.loadtxt(f)
    plt.plot(data[:,0]+8,data[:,1],'k',label='Typical atmos.')
    plt.legend()
    plt.savefig(f"{figdir}/temp_vs_z.png")

    plt.figure(2)
    for k in np.arange(nz-1, -1, -1):
        if k in k_to_delete:
            continue
        plt.plot(np.mean(pressure_3d[:,:,k]), np.mean(z_grid[:,:,k]), 'xk', label="mean" if k == 0 else "")
        plt.plot(np.max( pressure_3d[:,:,k]), np.mean(z_grid[:,:,k]), 'or', label="max"  if k == 0 else "")
        plt.plot(np.min( pressure_3d[:,:,k]), np.mean(z_grid[:,:,k]), '^b', label="min"  if k == 0 else "")
    plt.ylim([0, 20000])
    plt.xlabel('p (mbar)',fontsize=15)
    plt.ylabel(r'$z$ (m)',fontsize=15)

    with resources.open_text('erftools.data.typical_atmosphere', 'pressure_vs_z_actual.txt') as f:
        data = np.loadtxt(f)
    plt.plot(data[:,0],data[:,1],'k',label='Typical atmos.')
    plt.legend()
    plt.savefig(f"{figdir}/pressure_vs_z.png")

    plt.figure(3)
    for k in np.arange(nz-1, -1, -1):
        if k in k_to_delete:
            continue
        plt.plot(np.mean(theta_3d[:,:,k]), np.mean(z_grid[:,:,k]), 'xk', label="mean" if k == 0 else "")
        plt.plot(np.max( theta_3d[:,:,k]), np.mean(z_grid[:,:,k]), 'or', label="max"  if k == 0 else "")
        plt.plot(np.min( theta_3d[:,:,k]), np.mean(z_grid[:,:,k]), '^b', label="min"  if k == 0 else "")
    plt.ylim([0, 20000])
    plt.xlim([280, 600])
    plt.xlabel(r'$\theta$ (K)',fontsize=15)
    plt.ylabel(r'$z$ (m)',fontsize=15)

    with resources.open_text('erftools.data.typical_atmosphere', 'theta_vs_z_actual.txt') as f:
        data = np.loadtxt(f)
    plt.plot(data[:,0],data[:,1],'k',label='Typical atmos.')
    plt.legend()
    plt.savefig(f"{figdir}/theta_vs_z.png")

    plt.figure(4)
    for k in np.arange(nz-1, -1, -1):
        if k in k_to_delete:
            continue
        plt.plot(np.mean(qv_3d[:,:,k]), np.mean(z_grid[:,:,k]), 'xk', label="mean" if k == 0 else "")
        plt.plot(np.max( qv_3d[:,:,k]), np.mean(z_grid[:,:,k]), 'or', label="max"  if k == 0 else "")
        plt.plot(np.min( qv_3d[:,:,k]), np.mean(z_grid[:,:,k]), '^b', label="min"  if k == 0 else "")

    plt.ylim([0, 20000])
    plt.xlabel(r'$q_v$ (kg/kg)',fontsize=15)
    plt.ylabel(r'$z$ (m)',fontsize=15)
    plt.legend()
    plt.savefig(f"{figdir}/qv_vs_z.png")

    plt.figure(5)
    for k in np.arange(nz-1, -1, -1):
        if k in k_to_delete:
            continue
        plt.plot(np.mean(qsat_3d[:,:,k]), np.mean(z_grid[:,:,k]), 'xk', label="mean" if k == 0 else "")
        plt.plot(np.max( qsat_3d[:,:,k]), np.mean(z_grid[:,:,k]), 'or', label="max"  if k == 0 else "")
        plt.plot(np.min( qsat_3d[:,:,k]), np.mean(z_grid[:,:,k]), '^b', label="min"  if k == 0 else "")

    plt.xlabel(r'$q_{sat}$ (kg/kg)',fontsize=15)
    plt.ylabel(r'$z$ (m)',fontsize=15)
    plt.legend()
    plt.ylim([0, 20000])
    plt.xlim([0, 0.03])
    plt.savefig(f"{figdir}/qsat_vs_z.png")
