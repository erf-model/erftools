# for reference, see WRF/run/README.namelist

pbl_mapping = {
    0:  'None',
    1:  'YSU', # YSU scheme
    2:  'MYJ', # Mellor-Yamada-Janjic TKE scheme
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
    1: 'SurfaceLayer', # Revised MM5 Monin-Obukhov scheme (Jimenez, renamed in v3.6)
    2: 'SurfaceLayer', # Monin-Obukhov (Janjic) scheme
    5: 'SurfaceLayer', # MYNN surface layer
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
    2: 'SAM',       # SAM is comparable to Lin et al. scheme
    3: 'SAM',       # WRF Single-Moment, 3-class scheme with ice
    4: 'SAM',       # WRF Single-Moment, 5-class scheme
    5: 'SAM',       # Ferrier (new Eta), operational High-Resolution Window version
    6: 'SAM',       # WRF Single-Moment, 6-class scheme
    8: 'Morrison',  # Thompson scheme
    10: 'Morrison', # Morrison (double moment) scheme
    14: 'Morrison', # WRF Double-Moment 5-class scheme
    16: 'Morrison', # WRF Double-Moment 6-class scheme
    28: 'Morrison', # aerosol-aware Thompson scheme
}

ra_physics_mapping = {
    0: 'None',
    4: 'RRTMGP',
}

surface_physics_mapping = {
    0: 'None',
    1: 'SLM',       # thermal diffusion scheme (soil temperature only, 5 layers)
    2: 'NoahMP',    # Unified Noah land-surface model
    4: 'NoahMP',    # Noah-MP land-surface model
}

cu_physics_mapping = {
    0: 'None',
}

adv_mapping = {
    2: 'Centered_2nd',
    3: 'Upwind_3rd',
    4: 'Centered_4th',
    5: 'Upwind_5th',
    6: 'Centered_6th',
}

moist_adv_mapping = {
    0: 'simple',
    1: 'positive definite',
    2: 'monotonic',
    3: 'WENO5',
}

diff_opt_mapping = {
    1: 'simple',
    2: 'full',
}

km_opt_mapping = {
    1: 'constant',
    2: 'Deardorff', # 1.5 order TKE closure (3D)
    3: 'Smagorinsky', # Smagorinsky first-order closure (3D)
    4: 'Smagorinsky2D', # horizontal Smagorinsky closure (diagnosed from just
                        #   horizontal deformation); vertical diffusion from
                        #   PBL scheme
}

damp_opt_mapping = {
    0: 'none',
    1: 'increased diffusion',
    2: 'Rayleigh relaxation',
    3: 'Rayleigh implicit gravity-wave damping',
}
