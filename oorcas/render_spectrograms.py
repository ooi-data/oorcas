"""given a start and end day, render a daily spectrogram for each day and 24 hourly spectrograms
for each day. Save spectrograms to date directory on runner machine and syn with OOI
rca s3 bucket."""

import os
from datetime import datetime
from loguru import logger
import matplotlib.pyplot as plt
# we need to get ooipy 1.2 up on pypi!!
import ooipy

BASE_DIR = './spectrograms'

def get_s3_kwargs():
    aws_key = os.environ.get("AWS_KEY")
    aws_secret = os.environ.get("AWS_SECRET")
    
    s3_kwargs = {'key': aws_key, 'secret': aws_secret}
    return s3_kwargs


def plot_xr_spectrogram(spec):
    fig, ax = plt.subplots(figsize=(18, 6))
    c = ax.contourf(spec.time, spec.frequency, spec.T, levels=300, cmap='Spectral_r', vmin=45, vmax=90)
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    ax.set_yscale('log')
    ax.set_ylim(10, 32000)
    fig.colorbar(c, ax=ax, label="dB rel µ Pa^2 / Hz")
    return fig


def render_daily_bb_spectrogram(refdes, spec_date, gapless_merge, end_date=None,):
    s3_kwargs = get_s3_kwargs()

    spec_date = datetime.strptime(spec_date, "%Y-%m-%d")
    year_dir = os.path.join(BASE_DIR, spec_date.year)
    month_dir = os.path.join(year_dir, spec_date.month)
    day_dir = os.path.join(month_dir, spec_date.day)
    os.makedirs(day_dir, exist_ok=True)

    node = refdes[9:14] # ooipy uses node to index hydrophones, may change in future
    # get list of days to render
    hdata = ooipy.get_acoustic_data(spec_date, end_date, node, verbose=True, gapless_merge=gapless_merge)
    
    hourly_spec = hdata.compute_spectrogram(avg_time = 0.05, L=2048) #hardcoded for now
    daily_spec = hdata.compute_spectrogram(avg_time = 1, L=2048) #hardcoded for now

    daily_plot = plot_xr_spectrogram(daily_spec)
    filename = os.path.join(day_dir, f"{refdes}_{spec_date.strftime('%Y_%m_%d')}_day_spectrogram.png")
    daily_plot.savefig(filename, format='png')


    # start_date = datetime.strptime(start_date, "%Y-%m-%d")
    # if end_date:
    #     end_date = datetime.strptime(end_date, "%Y-%m-%d")