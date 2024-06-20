# vim: ts=4:sts=4:sw=4
#
# @author <lucile.gaultier@oceandatalab.com>
# @date 2024-01-10
#
# Copyright (C) 2024 OceanDataLab

"""
Create Odysea L3 like data from netcdf files and a python parameter file
"""


from odysim.swath_sampling import OdyseaSwath

import numpy
import os
import sys
import tqdm
import matplotlib
import datetime
import itertools
import xarray
from scipy.interpolate import RegularGridInterpolator
from typing import Optional, Tuple

# from dask.distributed import Client
# c = Client(n_workers=os.cpu_count()-2, threads_per_worker=1)

font = {'weight': 'bold',
        'size': 16}
matplotlib.rc('font', **font)

matplotlib.rc('font', **font)
matplotlib.rc('lines', linewidth=4)
matplotlib.rc('text', usetex=False)

matplotlib.rcParams.update({"axes.grid": True, "grid.color": "black"})


def load_python_file(file_path: str):
    """Load a file and parse it as a Python module."""
    if not os.path.exists(file_path):
        raise IOError('File not found: {}'.format(file_path))

    full_path = os.path.abspath(file_path)
    python_filename = os.path.basename(full_path)
    module_name, _ = os.path.splitext(python_filename)
    module_dir = os.path.dirname(full_path)
    if module_dir not in sys.path:
        sys.path.append(module_dir)

    module = __import__(module_name, globals(), locals(), [], 0)
    init_parameters(module)
    return module


def init_parameters(params):
    params.wind_path = getattr(params, 'wind_path', None)
    params.wind_speed = getattr(params, 'wind_speed', 7)
    params.wind_dir = getattr(params, 'wind_dir', 0)
    params.var_wind = getattr(params, 'var_wind', ('geo5_u10m', 'geo5_v10m'))
    params.var_current = getattr(params, 'var_current', ('SSU', 'SSV'))
    params.dic_coord = getattr(params, 'dic_coord', {})
    return None


def vradialSTDLookup(wind_speed: numpy.ndarray, wind_dir: numpy.ndarray,
                     encoder_angle: numpy.ndarray, azimuth: numpy.ndarray,
                     vradial_interpolator) -> numpy.ndarray:
    relative_azimuth = numpy.mod((numpy.mod(wind_dir + 180, 360) - 180
                                  - numpy.mod(azimuth +360, 360)) + 180, 360) - 180
    encoder_norm = numpy.mod(encoder_angle + 180, 360) - 180
    wind_speed[wind_speed>19] = 19
    interp = vradial_interpolator((wind_speed.flatten(),
                                   relative_azimuth.flatten(),
                                   encoder_norm.flatten()))
    return numpy.reshape(interp, numpy.shape(encoder_norm))


def generate_interpolator(lut_fn: str, key: Optional[str] = 'sigma_vr'
                          ) -> RegularGridInterpolator:
    lut = numpy.load(lut_fn)
    if 'wind_speed_range' in lut.files:
        wind_speed_range = lut['wind_speed_range']
    else:
        wind_speed_range = numpy.arange(0, 20, .1)
    if 'wind_dir_range' in lut.files:
        wind_dir_range = lut['wind_dir_range']
    else:
        wind_dir_range = numpy.arange(-195, 195, 5)
    if 'encoder_angle_range' in lut.files:
        encoder_angle_range = lut['encoder_angle_range']
    else:
        encoder_angle_range = numpy.arange(-190, 190, 5)
    vradial_lut = lut[key]
    vradial_interpolator = RegularGridInterpolator((wind_speed_range,
                                                    wind_dir_range,
                                                    encoder_angle_range),
                                                    vradial_lut,
                                                    bounds_error=False,
                                                    fill_value=0)
    return vradial_interpolator


def colocateSwathCurrents(model: xarray.Dataset, orbit: xarray.Dataset,
                          varu: str, varv:str) -> xarray.Dataset:

    """
    Colocate model current data to a swath (2d continuous array)
     of lat/lon/time query points.
    Ensure that lat/lon/time points of query exist within the loaded model
     data.

    Args:
        orbit (object): xarray dataset orbit generated via the
         orbit.getOrbit() call.
    Returns:
       original orbit containing model data linearly interpolated to
        the orbit swath. new data is contained in u_model, v_model

    """
    lats = orbit['lat'].values.flatten()
    lons = orbit['lon'].values.flatten()
    times = orbit['sample_time'].values.flatten()
    ds_u = model[varu].interp(time=xarray.DataArray(times, dims='z'),
                          lat=xarray.DataArray(lats, dims='z'),
                          lon=xarray.DataArray(lons, dims='z'),
                          method='linear')
    ds_v = model[varv].interp(time=xarray.DataArray(times, dims='z'),
                          lat=xarray.DataArray(lats, dims='z'),
                          lon=xarray.DataArray(lons, dims='z'),
                          method='linear')

    u_interp = numpy.reshape(ds_u.values, numpy.shape(orbit['lat'].values))
    v_interp = numpy.reshape(ds_v.values, numpy.shape(orbit['lat'].values))

    orbit = orbit.assign({'u_model': (['along_track', 'cross_track'], u_interp),
                          'v_model': (['along_track', 'cross_track'], v_interp)})

    return orbit


def load_orbit(orbit_fname: str, config_fname: str,
               start_time: datetime.datetime, end_time: datetime.datetime,
               year_ref: Optional[int] = 2020) -> xarray.Dataset:
    odysea = OdyseaSwath(orbit_fname=orbit_fname, config_fname=config_fname,
                         year_ref=year_ref)
    orbits = odysea.getOrbits(start_time=start_time, end_time=end_time)
    return orbits


def load_model(path_model: str, start: datetime.datetime,
               end: datetime.datetime, dic_coord: Optional[dict] = {}
               ) -> xarray.Dataset:

    model = xarray.open_mfdataset(path_model, combine='by_coords')
    #model.time.values.astype(float)
    if 'time_units' in dic_coord.keys():
        attrs = {'units': dic_coord['time_units']} #'days since 1950-01-01'}
        time = xarray.Dataset({'time': ('time', model.time.values, attrs)})
        time = xarray.decode_cf(time)
        model['time'] = time['time'].astype('datetime64[ns]')
        _ = dic_coord.pop('time_units')

    strstart = datetime.datetime.strftime(start, '%Y-%m-%d')
    strend = datetime.datetime.strftime(end, '%Y-%m-%d')
    model = model.sel(time=slice(strstart, strend))
    if len(dic_coord.keys()) > 0:
        model = model.rename(name_dict=dic_coord)
    return model


def convert_wind(model: xarray.Dataset, varu: Optional[str] = 'u_model',
                 varv: Optional[str] = 'v_model'):
    model['norm'] = numpy.sqrt(model[varu]**2 + model[varv]**2)
    model['direction'] = numpy.arctan2(model[varu]/model['norm'],
                                     model[varv]/model['norm'])
    model['direction'] = numpy.rad2deg(model['direction'])
    return model


def interp_model(o: xarray.Dataset, model: xarray.Dataset, bb: list,
                 varu: str, varv:str, wind: Optional[bool] = False,
                 asc: Optional[bool] = True) -> xarray.Dataset:
    if bb[1] > 180:
        o['lon'] = numpy.mod(o['lon'] + 360, 360)
        model[var_lon] = numpy.mod(model[var_lon] + 360, 360)
    if asc is True:
        _slice = slice(0, int(o.along_track.shape[0]/2))
        ind0 = -1
        ind1 = 0
    else:
        _slice = slice(int(o.along_track.shape[0]/2), o.along_track.shape[0])
        ind0 = -1
        ind1 = 0
    otmp = o.isel(along_track=_slice)
    iminmax = numpy.where((otmp.lat.data[:, ind1] >= bb[2])
                          & (otmp.lat.data[:, ind0] <= bb[3]))[0]
    print(iminmax[0], iminmax[-1])
    if asc is True:
        o2 = otmp.isel(along_track=slice(iminmax[0], iminmax[-1]))
    else:
        o2 = otmp.isel(along_track=slice(iminmax[0], iminmax[-1]))
    if numpy.max(o2.lon) < bb[0]:
        return None
    if numpy.min(o2.lon) > bb[1]:
        return None
    if numpy.min(o2.lon) == -180 and numpy.max(o2.lon) == 180:
        return None
    o2 = colocateSwathCurrents(model, o2, varu, varv)
    if wind is True:
        return o2
    o2['creator_name'] = 'Alexander Wineteer, Lucile Gaultier'
    o2['institution'] = 'JPL, ODL'
    alpha_fore = numpy.deg2rad(o2.azimuth_fore) - numpy.pi / 2
    alpha_aft = numpy.deg2rad(o2.azimuth_aft) - numpy.pi / 2
    ur_fore = (o2.u_model * numpy.cos(alpha_fore)
               + o2.v_model * numpy.sin(alpha_fore))
    ur_aft = (o2.u_model * numpy.cos(alpha_aft)
              + o2.v_model * numpy.sin(alpha_aft))
    alpha_fore = (numpy.rad2deg(alpha_fore) + 360) % 360
    alpha_aft = (numpy.rad2deg(alpha_aft) + 360) % 360
    alpha_fore = (alpha_fore + 180) % 360 - 180
    alpha_aft = (alpha_aft + 180) % 360 - 180
    o2 = o2.assign({'ur_nonoise_fore': (['along_track', 'cross_track'], ur_fore.data),
                    'ur_nonoise_aft': (['along_track', 'cross_track'], ur_aft.data),
                    'radial_angle_fore': (['along_track', 'cross_track'], alpha_fore.data),
                    'radial_angle_aft': (['along_track', 'cross_track'], alpha_aft.data)})
    o2['lon'] = numpy.mod(o['lon'] + 180, 360) - 180
    return o2


def generate_error(wind_speed: numpy.ndarray, wind_dir: numpy.ndarray,
                   encoder_fore: numpy.ndarray, encoder_aft: numpy.ndarray,
                   azimuth_fore: numpy.ndarray, azimuth_aft: numpy.ndarray,
                   vradial_interpolator: RegularGridInterpolator
                   ) -> Tuple[numpy.ndarray, numpy.ndarray]:
    std_fore = vradialSTDLookup(wind_speed, wind_dir, encoder_fore,
                                azimuth_fore, vradial_interpolator)
    std_aft = vradialSTDLookup(wind_speed, wind_dir, encoder_aft,
                               azimuth_aft, vradial_interpolator)

    # These are the results to be added to the radial current fore and aft
    err_fore = numpy.random.normal(scale=std_fore)
    err_aft = numpy.random.normal(scale=std_aft)
    return err_fore, err_aft


def error_on_swath(o, vradial_interpolator,
                   wind: Optional[xarray.Dataset] = None,
                   wind_norm: Optional[float] = 7,
                   wind_dir: Optional[float] = 0) -> xarray.Dataset:
    if wind is None:
        wind_speed = wind_norm * numpy.ones(numpy.shape(o['encoder_fore']))
        wind_dir = wind_dir * numpy.zeros(numpy.shape(o['encoder_fore']))
    else:
        wind_speed = wind['norm'].data
        wind_dir = wind['direction'].data
    err_fore, err_aft = generate_error(wind_speed, wind_dir,
                                       o['encoder_fore'].data,
                                       o['encoder_aft'].data,
                                       o['azimuth_fore'].data,
                                       o['azimuth_aft'].data,
                                       vradial_interpolator)
    ur_fore = o['ur_nonoise_fore'] + err_fore
    ur_aft = o['ur_nonoise_aft'] + err_aft
    o = o.assign({'ur_fore': (['along_track', 'cross_track'], ur_fore.data),
                  'ur_aft': (['along_track', 'cross_track'], ur_aft.data),
                  'err_fore': (['along_track', 'cross_track'], err_fore.data),
                  'err_aft': (['along_track', 'cross_track'], err_aft.data),
                  'wind_speed': (['along_track', 'cross_track'], wind_speed.data),
                  'wind_dir': (['along_track', 'cross_track'], wind_dir.data),
                  })
    return o


def job_odysea(o, model, i, c, bb, pattern_out, asc):
    oa_out = interp_model(o, model, bb, asc=True)
    if oa_out is not None:
        oa_out.to_netcdf(f'{pattern_out}_c_{c:02d}_p{i:03d}.nc', 'w')


def alac2xy(uac: numpy.ndarray, ual: numpy.ndarray, angle: numpy.ndarray,
            signu: float) -> Tuple[numpy.ndarray, numpy.ndarray]:
    ux = signu * (uac * numpy.cos(angle) + ual * numpy.cos(angle + numpy.pi/2))
    uy = signu * (uac * numpy.sin(angle) + ual * numpy.sin(angle + numpy.pi/2))
    return ux, uy


def xy2alac(ux: numpy.ndarray, uy: numpy.ndarray, angle: numpy.ndarray,
            signu: float) -> Tuple[numpy.ndarray, numpy.ndarray]:
    uac = signu * (ux * numpy.cos(angle) + uy * numpy.sin(angle))
    ual = signu * (-ux * numpy.sin(angle) + uy * numpy.cos(angle))
    return uac, ual


def angle_across(lon, lat) -> numpy.ndarray:
    angle = numpy.full(numpy.shape(lon), fill_value=numpy.nan)
    dlon = lon[:, 1:] - lon[:, :-1]
    dlat = lat[:, 1:] - lat[:, :-1]
    coslat = numpy.cos(numpy.deg2rad(lat[:, 1:]))
    angle[:, :-1] = numpy.angle(dlon * coslat + 1j * dlat)
    angle[:, -1] = angle[:, -2]
    return angle


def make_oi(o: xarray.Dataset, signu: Optional[float] = 1) -> xarray.Dataset:
    import optimal_interpolation as oi
    dic_in = {}
    list_key = ('radial_angle_fore', 'radial_angle_aft', 'ur_fore', 'ur_aft',
                'ur_nonoise_aft', 'ur_nonoise_fore', 'along_track', 'err_fore',
                'err_aft',
                'cross_track')
    for key in list_key:
        dic_in[key] = o[key].data
    angle = angle_across(o['lon'].data, o['lat'].data)
    dic_out = oi.perform_oi_on_l3(dic_in, ('ur', 'ur_nonoise', 'err'))

    # l2c_dic[key]['ac'][numpy.abs(grd['ac2']) < ac_thresh] = numpy.nan
    for key in ('ur', 'ur_nonoise', 'err'):
        u_out = alac2xy(dic_out[f'{key}_ac'], dic_out[f'{key}_al'], angle, signu)
        dic_out[f'{key}_eastward'], dic_out[f'{key}_northward'] = u_out
    u_out = xy2alac(o['u_model'].data, o['v_model'].data, angle, signu)
    dic_out['u_ac_model'], dic_out['u_al_model'] = u_out
    for key in dic_out.keys():
        o = o.assign({key: (['along_track', 'cross_track'], dic_out[key])})
    return o


def make_uv(o: xarray.Dataset, signu: Optional[float] = 1) -> xarray.Dataset:
    dic_in = {}
    dic_out = {}
    list_key = ('radial_angle_fore', 'radial_angle_aft', 'ur_fore', 'ur_aft',
                'ur_nonoise_aft', 'ur_nonoise_fore', 'along_track', 'err_fore',
                'err_aft', 'encoder_aft', 'encoder_fore',
                'cross_track')
    list_radial = ('ur', 'ur_nonoise', 'err')
    for key in list_key:
        dic_in[key] = o[key].data
    for key in list_radial:
        norm = numpy.sqrt(dic_in[f'{key}_fore']**2 + dic_in[f'{key}_aft']**2)
        enc_angle = numpy.pi * 1 / 2 + numpy.deg2rad(dic_in['encoder_aft'])
        dic_out[f'{key}_al'] = norm /2 /numpy.cos(enc_angle)
        dic_out[f'{key}_ac'] = norm /2 /numpy.sin(enc_angle)

    angle = angle_across(o['lon'].data, o['lat'].data)
    # l2c_dic[key]['ac'][numpy.abs(grd['ac2']) < ac_thresh] = numpy.nan
    for key in list_radial:
        u_out = alac2xy(dic_out[f'{key}_ac'], dic_out[f'{key}_al'], angle, signu)
        dic_out[f'{key}_northward'], dic_out[f'{key}_eastward'] = u_out
    u_out = xy2alac(o['u_model'].data, o['v_model'].data, angle, signu)
    dic_out['u_ac_model'], dic_out['u_al_model'] = u_out
    for key in dic_out.keys():
        o = o.assign({key: (['along_track', 'cross_track'], dic_out[key])})
    return o

def generate_pass(params, i: int, c: int, o: xarray.Dataset,
                  model: xarray.Dataset, wind_data: xarray.Dataset,
                  vradial_interpolator: RegularGridInterpolator,
                  var_current: list, var_wind: list,
                  asc: Optional[bool] = True) -> None:
    o_out = interp_model(o, model, params.bounding_box, var_current[0],
                         var_current[1], wind=False, asc=asc)
    if asc is True:
        signu = 1
    else:
        signu = -1
    wind_o = None
    if o_out is not None:
        if wind_data is not None:
            ow = o.copy()
            wind_o = interp_model(ow, wind_data, params.bounding_box,
                                  var_wind[0], var_wind[1],
                                  asc=asc, wind=True)
            wind_o = convert_wind(wind_o)
        o_out = error_on_swath(o_out, vradial_interpolator,
                               wind=wind_o, wind_norm=params.wind_speed,
                               wind_dir=params.wind_dir)
        #o_out = make_uv(o_out, signu=signu)
        o_out = make_oi(o_out, signu=signu)
        file_out = os.path.join(params.path_out,
                                f'{params.pattern_out}_c{c:02d}_p{i:03d}.nc')
        o_out.to_netcdf(file_out, 'w')


if __name__ == '__main__':
    conf = 'confE2'
    conf = 'confE2ibi'
    params = load_python_file(f'params_{conf}.py')
    orbits = load_orbit(params.orbit_file, params.config_file,
                        params.start_time, params.end_time,
                        year_ref=params.year_ref)
    model = load_model(params.path_model, params.start_time, params.end_time,
                       dic_coord=params.dic_coord)
    wind_data = None
    if params.wind_path is not None:
        wind_data = load_model(params.path_wind, params.start_time,
                               params.end_time, dic_coord=params.dic_coord)
    vradial_interpolator = generate_interpolator(params.lut_fn,
                                                 key=params.sigma_vr)
    os.makedirs(params.path_out, exist_ok=True)
    i = 0
    c = 1
    for o in tqdm.tqdm(itertools.islice(orbits, 400)):
        i += 1
        generate_pass(params, i, c, o, model, wind_data, vradial_interpolator,
                      params.var_current, params.var_wind,
                      asc=True)
        i += 1
        generate_pass(params, i, c, o, model, wind_data, vradial_interpolator,
                      params.var_current, params.var_wind,
                      asc=False)
        if i > 999:
            c = c + 1
            i = 0
