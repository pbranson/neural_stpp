# Copyright (c) Facebook, Inc. and its affiliates.

import pandas as pd
import numpy as np
import urllib.request
import shutil
import os
import xarray as xr


def preprocess():

    os.makedirs("earthquakes", exist_ok=True)

    ds = xr.open_zarr('test_partitions_chunked_pmt3.zarr.zip',chunks='auto').isel(site=0)

    basedate = pd.to_datetime(pd.to_datetime(ds.time[0].values).date())
    ds['time'] = (ds.time - basedate.asm8)/pd.to_timedelta(1,'d')

    ds_events = ds.stack(event=['time','part'])
    ds_events = ds_events.where(ds_events.hs > 0.1, drop=True)
    ds_events['dp_rad'] = np.deg2rad(ds_events.dp + 7.5/2*np.random.randn(*ds_events.dp.shape))
    ds_events['hs_u'], ds_events['hs_v'] = ds_events.hs * np.sin(ds_events.dp_rad), ds_events.hs * np.cos(ds_events.dp_rad),

    ds_features = ds_events.reset_index('event').reset_coords()[['time','hs', 'tp']]

    df = ds_features.to_pandas().reset_index().drop(columns='event')

    df = df.set_index('time')
    print(len(df))
    df = df.dropna()
    print(len(df))

    sequences = {}
    for weeks in range(52 * 30 + 1):
        date = basedate + pd.Timedelta(weeks=weeks)

        seq_name = f"{date.year}{date.month:02d}" + f"{date.day:02d}"

        start = (date - basedate).days
        end = (date + pd.Timedelta(days=7) - basedate).days

        df_range = df[start+1/24:end-1/24]
        df_range.index = df_range.index - start

        t = df_range.index + np.random.uniform(-1,1,size=df_range.index.shape) * 0.5/24
        t = t.values
        sort_inds = t.argsort()
        t = t[sort_inds]
        x = df_range.to_numpy().astype(np.float64)[sort_inds]
        print(len(t))

        sequences[seq_name] = np.concatenate([t[:,None],x],axis=1)

    np.savez("swells/swells_rottnest.npz", **sequences)
    print("Preprocessing complete.")


if __name__ == "__main__":
    
    preprocess()
