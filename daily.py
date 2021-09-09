from datetime import datetime, timedelta
import functools
import os
from typing import Callable, Dict, Hashable

from bs4 import BeautifulSoup
from dask.diagnostics import ProgressBar
import fsspec
from fsspec.implementations.local import LocalFileSystem
from fsspec_reference_maker.hdf import SingleHdf5ToZarr
import pandas as pd
from pangeo_forge_recipes.patterns import pattern_from_file_sequence, FilePattern, Index
from pangeo_forge_recipes.recipes import HDFReferenceRecipe
from pangeo_forge_recipes.recipes.reference_hdf_zarr import ChunkKey, _unstrip_protocol
from pangeo_forge_recipes.storage import FSSpecTarget, MetadataTarget
import requests


START_DATE = "1981-09-01"
# START_DATE = "2018-12-01"


def current_downloads():
    """
    get the current year and month and the previous year and month as zero padded strings.
      Only really matters in January.
    for use in the OISST_DAILY_FILES download script when we need today's year, month and the previous year month
    returns a list of strings [year, month, prev_year, prev_month]
    E.g. ["2019", "01", "2018", "12"]
    """
    # To test other dates
    #  today = datetime.date(2018, 10, 1)

    today = datetime.today()
    first = today.replace(day=1)
    prev_month = first - timedelta(days=1)
    # convert to zero padded strings
    year = str(today.year)
    month = "{0:02d}".format(today.month)
    prev_year = str(prev_month.year)
    prev_month = "{0:02d}".format(prev_month.month)
    return (year, month, prev_year, prev_month)


def netcdf_links(year: str, month: str) -> list[str]:
    """ Return any netcdf links for the selected month and year """
    fetch_url = f"https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/{year}{month}/"
    req = requests.get(fetch_url)

    soup = BeautifulSoup(req.text, "html.parser")
    links = []

    for link in soup.find_all("a"):
        href = link.get("href")
        if ".nc" in href:
            links.append(fetch_url + href)
    return links


class CachedHDFReferenceRecipe(HDFReferenceRecipe):
    @property
    def store_chunk(self) -> Callable[[Hashable], None]:
        return functools.partial(
            _one_chunk,
            file_pattern=self.file_pattern,
            netcdf_storage_options=self.netcdf_storage_options,
            metadata_cache=self.metadata_cache,
        )


def _one_chunk(
    chunk_key: ChunkKey,
    file_pattern: FilePattern,
    netcdf_storage_options: Dict,
    metadata_cache: MetadataTarget,
):
    fname = file_pattern[chunk_key]
    fn = os.path.basename(fname + ".json")

    if not metadata_cache.exists(fn):
        with fsspec.open(fname, **netcdf_storage_options) as f:

            h5chunks = SingleHdf5ToZarr(
                f, _unstrip_protocol(fname, f.fs), inline_threshold=300
            )
            metadata_cache[fn] = h5chunks.translate()


if __name__ == "__main__":
    year, month, prev_year, prev_month = current_downloads()

    print("Finding recent OISST NetCDF URLs to find extent of preliminary data")

    urls = netcdf_links(prev_year, prev_month) + netcdf_links(year, month)
    prelim_urls = [url for url in urls if "preliminary" in url]
    complete_urls = [url for url in urls if "preliminary" not in url]

    print("Generating preliminary OISST recipe")

    prelim_pattern = pattern_from_file_sequence(prelim_urls, "time", nitems_per_file=1)

    fs_local = LocalFileSystem()

    prelim_cache_dir = "./cache/preliminary/"
    prelim_metadata_cache = MetadataTarget(fs_local, prelim_cache_dir)

    print("Cleaning up preliminary cache")

    prelim_dates = {
        url.split("/")[-1].split(".")[1].split("_")[0] for url in prelim_urls
    }
    for key in prelim_metadata_cache.get_mapper().keys():
        key_date = key.split(".")[1].split("_")[0]
        if key_date not in prelim_dates:
            print(f"{key} is no longer a preliminary file, removing from cache")
            prelim_metadata_cache.rm(key)

    prelim_target_dir = "./preliminary/"
    prelim_target = FSSpecTarget(fs_local, prelim_target_dir)

    prelim_recipe = CachedHDFReferenceRecipe(
        prelim_pattern, metadata_cache=prelim_metadata_cache, target=prelim_target
    )

    delayed = prelim_recipe.to_dask()

    print("Calculating preliminary OISST recipe")

    with ProgressBar():
        delayed.compute()

    print("Generating complete OISST recipe")

    complete_latest_date = complete_urls[-1].split("/")[-1].split(".")[1]
    complete_latest_dt = datetime.strptime(complete_latest_date, "%Y%m%d")

    dates = pd.date_range(START_DATE, complete_latest_dt, freq="D")

    input_url_pattern = (
        "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation"
        "/v2.1/access/avhrr/{yyyymm}/oisst-avhrr-v02r01.{yyyymmdd}.nc"
    )
    input_urls = [
        input_url_pattern.format(
            yyyymm=day.strftime("%Y%m"), yyyymmdd=day.strftime("%Y%m%d")
        )
        for day in dates
    ]

    complete_pattern = pattern_from_file_sequence(input_urls, "time", nitems_per_file=1)

    complete_cache_dir = "./cache/complete/"
    complete_metadata_cache = MetadataTarget(fs_local, complete_cache_dir)

    complete_target_dir = "./complete/"
    complete_target = FSSpecTarget(fs_local, complete_target_dir)

    complete_recipe = CachedHDFReferenceRecipe(
        complete_pattern, metadata_cache=complete_metadata_cache, target=complete_target
    )

    delayed = complete_recipe.to_dask()

    print("Calculating complete OISST recipe")

    with ProgressBar():
        delayed.compute()

