# Copyright (c) Facebook, Inc. and its affiliates.

import pandas as pd
import numpy as np
import urllib.request
import shutil
import os
import xarray as xr

def get_fset(fset):
    if fset == 0:
        return ['hs','tpn']
    elif fset == 1:
        return ['hs_u','hs_v','tpn']
    elif fset == 2:
        return ['hs','dp_u','dp_v','tpn']
    elif fset == 3:
        return ['hs','dp_u','dp_v','tpn','gamma']
    elif fset == 4:
        return ['hs','tpn','gamma']
    elif fset == 5:
        return ['hs','tpn','t_u','t_v']

def preprocess(fset):

    features = get_fset(fset)

    os.makedirs("swells", exist_ok=True)
    outfile = f"swells/swells_rottnest_fset{fset}.npz"
    if not os.path.exists(outfile):

        ds = xr.open_zarr('test_partitions_pmt3.2.zarr.zip',chunks='auto').isel(site=0)

        basedate = pd.to_datetime(pd.to_datetime(ds.time[0].values).date())
        ds['time'] = (ds.time - basedate.asm8)/pd.to_timedelta(1,'d')

        ds_events = ds.stack(event=['time','part'])
        ds_events = ds_events.where(ds_events.hs > 0.1, drop=True)

        maxT = 28.571428298950195
        ds_events['dpn'] = ds_events.dp + 15*0.367*(np.random.randn(*ds_events.dp.shape))
        ds_events['dpn'] = ds_events['dpn'] % 360.
        ds_events['tpn'] = ds_events.tp + ds_events.tp/maxT*(np.random.randn(*ds_events.tp.shape))
        ds_events['gamma'] = ds_events.gamma.where(ds_events.gamma > 1, ds_events.gamma + 0.05*np.random.randn(*ds_events.gamma.shape))

        ds_events['dpn_rad'] = np.deg2rad(ds_events.dpn)
        ds_events['dp_u'], ds_events['dp_v'] = np.sin(ds_events.dpn_rad), np.cos(ds_events.dpn_rad),
        ds_events['hs_u'], ds_events['hs_v'] = ds_events.hs * ds_events.dp_u, ds_events.hs * ds_events.dp_v

        ds_features = ds_events.reset_index('event').reset_coords()[['time',] + features]

        df = ds_features.to_pandas().reset_index().drop(columns='event')

        df = df.set_index('time')
        df['t_u'] = np.sin(df.index.day_of_year/366*2.*np.pi)
        df['t_v'] = np.cos(df.index.day_of_year/366*2.*np.pi)
        print(len(df))
        df = df.dropna()
        print(len(df))

        sequences = {}
        for weeks in range(0,52 * 40,2):
            date = basedate + pd.Timedelta(weeks=weeks)

            seq_name = f"{date.year}{date.month:02d}" + f"{date.day:02d}"

            start = (date - basedate).days
            end = (date + pd.Timedelta(days=14) - basedate).days

            df_range = df[start+1/24:end-1/24]
            df_range.index = df_range.index - start

            t = df_range.index + np.random.uniform(-1,1,size=df_range.index.shape) * 0.5/24
            t = t.values
            sort_inds = t.argsort()
            t = t[sort_inds]
            x = df_range.to_numpy().astype(np.float64)[sort_inds]

            print(len(t))

            sequences[seq_name] = np.concatenate([t[:,None],x],axis=1)

        np.savez(outfile, **sequences)
    print("Preprocessing complete.")


if __name__ == "__main__":
    import sys

    preprocess(int(sys.argv[1]))
