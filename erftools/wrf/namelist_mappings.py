pbl_mapping = {
    0:  'None',
    1:  'YSU', # YSU scheme
#   2:  'MYJ', # Mellor-Yamada-Janjic TKE scheme
#   3:  'GFS', # Hybrid EDMF GFS scheme
#   4:  'QNSE', # Eddy-diffusivity Mass Flux, Quasi-Normal Scale Elimination PBL
    5:  'MYNN', # MYNN 2.5/2.6/3.0 level TKE scheme
#   7:  'ACM2',
#   8:  'BouLac',
#   9:  'UW',
#   10: 'TEMF',
#   11: 'Shin-Hong',
#   12: 'GBM',
#   99: 'MRF',
}

sfclay_mapping = {
    0: 'None',
    1: 'MOST', # Revised MM5 Monin-Obukhov scheme (Jimenez, renamed in v3.6)
    2: 'MOST', # Monin-Obukhov (Janjic) scheme
    5: 'MOST', # MYNN surface layer
}
valid_sfclay = {
    # for each PBL scheme, based on acceptable WRF inputs
    0:  list(range(100)),
    1:  [1],
    2:  [2],
    5:  [1,2,5,91],
    6:  [1,2,5,91],
    7:  [1,7,91],
    8:  [1,2,91],
    9:  [1,2,91],
    11: [1,91],
    12: [1,91],
    91: [1,91],
}

mp_physics_mapping = {
    0: 'None',
    1: 'Kessler',
    2: 'SAM', # comparable to Lin et al. scheme
    6: 'SAM', # WRF Single-Moment, 6-class scheme
    28: 'SAM', # aerosol-aware Thompson scheme
}

ra_physics_mapping = {
    0: 'None',
    4: 'RRTMGP',
}

cu_physics_mapping = {
    0: 'None',
}

diff_opt_mapping = {
    1: 'simple',
    2: 'full',
}

km_opt_mapping = {
    1: 'constant',
    2: 'Deardorff', # 1.5 order TKE closure (3D)
    3: 'Smagorinsky', # Smagorinsky first-order closure (3D)
    4: '2D Smagorinsky', # horizontal Smagorinsky closure (diagnosed from just
                         #   horizontal deformation); vertical diffusion from
                         #   PBL scheme
}

damp_opt_mapping = {
    0: 'none',
    1: 'increased diffusion',
    2: 'Rayleigh relaxation',
    3: 'Rayleigh implicit gravity-wave damping',
}
