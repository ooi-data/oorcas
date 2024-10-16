import obspy as obs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import fsspec
from datetime import timedelta, datetime
from ooipy.hydrophone.basic import HydrophoneData
from datetime import datetime
from obspy.core import UTCDateTime
from tqdm import tqdm

import multiprocessing as mp
import concurrent.futures

from loguru import logger


def _map_concurrency(func, iterator, args=(), max_workers=-1, verbose=False):
    # automatically set max_workers to 2x(available cores)
    if max_workers == -1:
        max_workers = 2 * mp.cpu_count()

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = {executor.submit(func, i, *args): i for i in iterator}
        # Disable progress bar
        is_disabled = not verbose
        for future in tqdm(
            concurrent.futures.as_completed(future_to_url), total=len(iterator), disable=is_disabled
        ):
            data = future.result()
            results.append(data)
    return results


class HydrophoneDay:

    def __init__(
        self,
        refdes,
        str_date,
        data=None,
        mseed_urls=None,
        clean_list=None,
        stream=None,
        spec=None,
    ):
        self.refdes = refdes
        self.date = datetime.strptime(str_date, "%Y/%m/%d")
        self.data = data
        self.mseed_urls = self.get_mseed_urls(str_date, refdes)
        self.clean_list=clean_list
        self.stream=stream
        self.spec=spec
        self.file_str = f"{self.refdes}_{self.date.strftime('%Y_%m_%d')}"


    def get_mseed_urls(self, day_str, refdes):
        """
        get URLs for a specific day from OOI raw data server
    
        Parameters
        ----------
        day_str : str
            date for which URLs are requested; format: yyyy/mm/dd,
            e.g. 2016/07/15
        node : str
            identifier or name of the hydrophone node
        verbose : bool
            print exceptions if True
    
        Returns
        -------
        ([str], str)
            list of URLs, each URL refers to one data file. If no data is
            available for specified date, None is returned.
        """
        base_url = "https://rawdata.oceanobservatories.org/files"
    
        mainurl = f"{base_url}/{refdes[0:8]}/{refdes[9:14]}/{refdes[15:27]}/{day_str}/"
    
        FS = fsspec.filesystem("http")
        print(mainurl)
    
        try:
            data_url_list = sorted(
                f["name"]
                for f in FS.ls(mainurl)
                if f["type"] == "file" and f["name"].endswith(".mseed")
            )
        except Exception as e:
            print("Client response: ", e)
            return None
    
        if not data_url_list:
            print("No Data Available for Specified Time")
            return None
    
        return data_url_list

    
    def read_and_repair_gaps(self, fill_value, method, wav_data_subtype):
        self.clean_list = _map_concurrency(
            func=self._deal_with_gaps_and_overlaps, 
            args=(fill_value, method, wav_data_subtype), 
            iterator=self.mseed_urls, verbose=False
        )

    
    def create_single_stream(self, fill_value, method):
    
        for st in self.clean_list:
            cs = st.copy()
            if not isinstance(self.stream, obs.Stream):
                self.stream = cs
            else:
                self.stream += cs
                
        logger.info("Merging to a single stream for the day")
        self.stream.merge(fill_value=fill_value, method=method)
        
        self._clean_day_edges(fill_value, method)
        
        self.data = self.stream[0].data # for convenient access to actual acoustic data
        


    def plot_spectrogram(self, avg_time, L, sel=None):
        vmin=45
        vmax=90

        hdata = HydrophoneData(data=self.stream[0].data, header=self.stream[0].stats, node=self.refdes[9:14])
        if self.spec is None: 
            self.spec = hdata.compute_spectrogram(avg_time=avg_time, L=L)
            spec = self.spec
        if sel:
            spec = self.spec.sel(time=slice(sel[0], sel[1]))
        if self.spec is not None and not sel:
            spec = self.spec
        
        fig, ax = plt.subplots(figsize=(18, 6))
        c = ax.contourf(spec.time, spec.frequency, spec.T, 
            levels=np.linspace(vmin, vmax, 300), extend='both', cmap='Spectral_r', 
            vmin=vmin, vmax=vmax,
            rasterized=True,
        )
        cbar = fig.colorbar(c, ax=ax, label="dB rel µ Pa^2 / Hz", ticks=np.linspace(vmin, vmax, 10), pad=0.01)
            
        ax.set_ylabel('frequency')
        ax.set_yscale('log')
        ax.set_ylim(10, 32000)

        if sel: 
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        else:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.set_xlim(self.date, self.date + timedelta(hours=23, minutes=59, seconds=59, microseconds=999999))

        return fig

            
    def _merge_by_timestamps(self, st):
        cs = st.copy()
        
        data = []
        for tr in cs:
            data.append(tr.data)
        data_cat = np.concatenate(data)
    
        stats = dict(cs[0].stats)
        stats["starttime"] = st[0].stats["starttime"]
        stats["endtime"] = st[-1].stats["endtime"]
        stats["npts"] = len(data_cat)
    
        cs = obs.Stream(traces=obs.Trace(data_cat, header=stats))
    
        return cs
        

    def _deal_with_gaps_and_overlaps(self, url, fill_value, method, wav_data_subtype):
        if wav_data_subtype not in ["PCM_32", "FLOAT"]:
            raise ValueError("Invalid wave data subtype. Please specify 'PCM_32' or 'FLOAT'")
        # first read in mseed
        if wav_data_subtype == "PCM_32":
            st = obs.read(url, apply_calib=False, dtype=np.int32)
        if wav_data_subtype == "FLOAT":
            st = obs.read(url, apply_calib=False, dtype=np.float64)
        
        
        trace_id = st[0].stats["starttime"]
        print("total traces before concatenation: " + str(len(st)), flush=True)
        # if 19.2 samples +- 640 then concat
        samples = 0
        for trace in st:
            samples += len(trace)
            
        if 19199360 <= samples <= 19200640: # CASE A just jitter, no true gaps <<<
            print(f"There are {samples} samples in this stream, Simply concatenating")
            cs = self._merge_by_timestamps(st)
            print("total traces after concatenation: " + str(len(cs)))
        else:
            print(f"{trace_id}: there are a unexpected number of samples in this file. Checking for large gaps:")
            gaps = st.get_gaps()
            st_contains_large_gap = False
            # loop checks for large gaps
            for gap in gaps:
                if abs(gap[6]) > 0.02: # TODO decide on threshold, the gaps 6th element is the gap length NEEDS TO BE ABSOLUTE VALUE
                    st_contains_large_gap = True
                    break
            
            if st_contains_large_gap: # CASE C - edge case - real gaps that should be filled using obspy fill_value and method of choice
                print(f"{trace_id}: there is a gap not caused by jitter. Using obspy method={method}, fill_value={str(fill_value)}")
                cs = st.merge(method=method, fill_value=fill_value)
                print("total trace after merge: " + str(len(cs)))
            else: # # CASE B shortened trace before divert with no real gaps 
                print(f"{trace_id}: This file is short but only contains jitter. Simply concatenating")
                cs = self._merge_by_timestamps(st)
                print("total traces after concatenation: " + str(len(cs)), flush=True)
        return cs


    def _clean_day_edges(self, fill_value, method):
        # we need to fill in the rest of the day if it has been diverted - there's probably a better way
        clean_end_time = UTCDateTime(self.date.year, self.date.month, self.date.day, 23, 59, 59, 99)
        clean_start_time = UTCDateTime(self.date.year, self.date.month, self.date.day, 0, 0, 0, 1)
        if self.stream[0].stats.endtime < clean_end_time:
            logger.warning(f"End of day: {self.stream[0].stats.endtime} - appears to be incomplete due to diversion - setting manually")
            end_header = self.stream[0].stats.copy()
            end_header['starttime'] = clean_end_time
            end_header['npts'] = 1
            end_trace = obs.Trace(data=np.array([0], dtype=np.float64), header=end_header)
            self.stream += end_trace
            logger.info("Adding dummy trace to day end")
            self.stream.merge(fill_value=fill_value, method=method)
        if self.stream[0].stats.starttime > clean_start_time:
            logger.warning(f"Start of day: {self.stream[0].stats.starttime} - appears to be incomplete due to diversion - setting manually")
            start_header = self.stream[0].stats.copy()
            start_header['starttime'] = clean_start_time
            start_header['npts'] = 1 
            start_trace = obs.Trace(data=np.array([0], dtype=np.float64), header=start_header)
            self.stream += start_trace
            logger.info("Adding dummy trace to day start")
            self.stream.merge(fill_value=fill_value, method=method)
        
