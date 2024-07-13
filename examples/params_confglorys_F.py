import os
import datetime

# Geometry
conf = 'F'
#workdir = '/home/lgaultier/src/odysea-science-simulator/odysim/'
workdir = '/mnt/data/CNES_odysea/'
orbit_file = os.path.join(workdir, 'odysea_ephemerides_2024-06-04',
                          'odysea-587km-97p704deg-430-MLTAN_2024_02_27_orb_ECEF.txt')
config_file = os.path.join(workdir, conf, f'wacm_sampling_config_Conf{conf}.py')

# Instrument
#lut_dir = os.path.join(workdir, 'uncertainty_tables')
lut_dir = os.path.join(workdir, conf)

wind_norm = 7
wind_dir = 0
wind_path = '/mnt/data_6t/glorys_2009/wind*.nc'
path_wind = '/mnt/data_6t/glorys_2009/wind*.nc'
var_wind = ('u10m', 'v10m')
dic_coord_wind = {'Longitude': 'lon', 'Latitude': 'lat', 'time_units': 'hours since 1900-01-01'}
if conf == 'JPL':
    lutfile = 'odysea_sigma_vr_lut_height590km_look52deg_swath1672km.npz'
    sigma_vr = 'sigma_vr'
elif conf == 'E2':
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
#dic_coord = {'longitude': 'lon', 'latitude': 'lat', 'time_units': 'seconds since 1970-01-01' }
var_current = ('SSU', 'SSV')
year_ref = 2009
start_time = datetime.datetime.strptime('2009-01-01:00','%Y-%m-%d:%H')
end_time = datetime.datetime.strptime('2010-01-01:00','%Y-%m-%d:%H')
path_model = '/mnt/data/glorys_2009/*.nc'
bounding_box = [-180, 180, -90, 90]

# Output path
pattern_out = f'odysea_l3_global_glorys_conf{conf}'
# pattern_out = 'odysea_tropical_pacific'
path_out = f'/mnt/data_6t/odysea_l3_global_glorys_{conf}'
