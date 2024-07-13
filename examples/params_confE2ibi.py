import os
import datetime

# Geometry
conf = 'E2'
#workdir = '/home/lgaultier/src/odysea-science-simulator/odysim/'
workdir = '/mnt/data/CNES_odysea/'
orbit_file = os.path.join(workdir, 'odysea_ephemerides_2024-06-04',
                          'odysea-800km-98p6deg-430-MLTAN_orb_ECEF.txt')
config_file = os.path.join(workdir, conf, f'wacm_sampling_config_Conf{conf}.py')

# Instrument
#lut_dir = os.path.join(workdir, 'uncertainty_tables')
lut_dir = os.path.join(workdir, conf)

wind_norm = 7
wind_dir = 0
wind_path = None #'/mnt/data_b/mitgcm/wind10m/*.nc'
# var_wind = ('geo5_u10m', 'geo5_v10m')
if conf == 'E2':
    lutfile = 'odysea_sigma_vr_5km_avg_lut_height800km_look41deg_swath1497km_CNES.npz'
    sigma_vr = 'sigma_vr_5km_projected'
elif conf == 'F':
    lutfile = 'odysea_sigma_vr_5km_avg_lut_height590km_look49deg_swath1486km_CNES.npz'
    sigma_vr = 'sigma_vr_5km_projected'
elif conf == 'G':
    lutfile = 'odysea_sigma_vr_5km_avg_lut_height8000km_look41deg_swath1497km_CNES.npz'
    sigma_vr = 'sigma_vr_5km_projected'
elif conf == 'ConfB':
    lutfile = 'odysea_sigma_vr_5km_avg_lut_height602km_look49deg_swath1531km_CNES.npz'
    sigma_vr = 'sigma_vr_5km_projected'

lut_fn = os.path.join(lut_dir, lutfile)

# Model 
var_current = ('u', 'v')
dic_coord = {'longitude': 'lon', 'latitude': 'lat', 'time_units': 'days since 1950-01-01'}
year_ref = 2014
start_time = datetime.datetime.strptime('2014-01-01:00','%Y-%m-%d:%H')
end_time = datetime.datetime.strptime('2014-01-03:12','%Y-%m-%d:%H')
path_model = '/mnt/data_b/wetransfer_ibi_2024-04-30_1214/IBI36F_12em_v_*nc'
bounding_box = [-20, 17, 26, 64]

# Output path
pattern_out = f'odysea_l3_ibi_conf{conf}'
# pattern_out = 'odysea_tropical_pacific'
path_out = f'/mnt/data_6t/odysea_l3_ibi_{conf}'
