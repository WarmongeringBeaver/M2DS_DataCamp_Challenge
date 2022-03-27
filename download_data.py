"""Fetcher for RAMP data stored in OSF
To adapt it for another challenge, change the CHALLENGE_NAME and upload
public/private data as `tar.gz` archives in dedicated OSF folders named after
the challenge.
"""
import tarfile
import argparse
from zlib import adler32
from pathlib import Path
from osfclient.api import OSF
from osfclient.exceptions import UnauthorizedException
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_cleaning import data_cleaning

LOCAL_DATA = Path(__file__).parent / "data"

CHALLENGE_NAME = 'covid_vaccine_challenge'
# you might choosing checking for the correct checksum, if not set
# data_checksum to None
RAMP_FOLDER_CONFIGURATION = {
    'public': dict(
        code='rw9cu', 
        file_name_1='data_1.csv', 
        url = 'https://www.dropbox.com/s/wv6fvff69yo0rcl/covid-stringency-index.csv?dl=1',
        file_name_2 = 'data_2.csv'
    ),
}

def load_clean_data(fname):
    df = pd.read_csv(fname, encoding='iso-8859-1', na_values= ' ')
    df = data_cleaning(df)
    return df

def get_connection_info():
    "Get connection to OSF and info relative to data."
    osf, folder_name = OSF(), 'public'
    data_config = RAMP_FOLDER_CONFIGURATION[folder_name]
    project = osf.project(data_config['code'])
    store = project.storage('osfstorage')
    return store, data_config


def get_data_from_osf(container):
    "Get data from OSF container with a comprehensible failure error."
    elements = [f for f in container]
    assert len(elements) == 1, (
        'There is no element in osf storage' 
    )
    return elements[0]


def download_from_osf():
    "Download the data from OSF."

    # check if data directory is empty
    if not LOCAL_DATA.exists() or not any(LOCAL_DATA.iterdir()):
        LOCAL_DATA.mkdir(exist_ok=True)

        print("Checking the data URL...")
        # Get the connection to OSF
        store, data_config = get_connection_info()

        # ----------------------------------------------------------------------
        # DATA SET 1 (MAIN)
        # ----------------------------------------------------------------------

        # Find the file to download from the OSF project
        challenge_file = get_data_from_osf(store.files)

        # Find the file to download from the OSF project
        file_name_1 = data_config['file_name_1']
        print('Ok.')

        # Download the file in the data
        FILE_PATH_1 = LOCAL_DATA / file_name_1
        print("Downloading the data...")
        with open(FILE_PATH_1, 'wb') as f:
            challenge_file.write_to(f)

        # Clean data (only for file 1)
        df1 = load_clean_data(FILE_PATH_1)

        # ----------------------------------------------------------------------
        # DATA SET 2 (STRINGENCY)
        # ----------------------------------------------------------------------
        
        # download second data set
        string_data = get_stringency(data_config['url'])
        file_name_2 = data_config['file_name_2']
        FILE_PATH_2 = LOCAL_DATA / file_name_2
        with open(FILE_PATH_2, 'wb') as f:
            f.write(string_data)

        df2 = pd.read_csv(FILE_PATH_2, encoding='iso-8859-1', na_values= ' ')

        # ----------------------------------------------------------------------
        # MERGE
        # ----------------------------------------------------------------------
        df2=df2.loc[(df2['Day'] == '2020-09-01') & (df2['Entity'].isin(['Brazil',
                                  'Canada', 'China', 'Ecuador', 'France',
                                  'Germany', 'India', 'Italy',
                                  'Mexico', 'Nigeria', 'Poland', 'Russia',
                                  'South Africa', 'Singapore', 'South Korea',
                                  'Spain', 'Sweden', 'United Kingdom','United States']))].copy()
        df2 = df2.drop(columns=['Code', 'Day'])
        df2['population']=[212559417,37742154,1439323776,17643054,65273511,83783942,1380004385,
                    60461826,128932753,206139589,37846611,145934462,59308690,5850342,51269185,
                    46754778,10099265,67886011,331002651]
        df2['Entity'] = df2['Entity'].replace(['United Kingdom','United States'],['UK', 'US'])
        
        df_merged=pd.merge(df1, df2, left_on='Country', right_on='Entity').drop(columns='Entity')

        df_merged['case_rate'] = (df_merged['covidcases']/df_merged['population'])*100
        df_merged['death_rate'] = (df_merged['coviddeaths']/df_merged['population'])*100

        # persist merged data set, remove separate ones
        FILE_PATH_1.unlink()
        FILE_PATH_2.unlink()
        FILE_PATH = LOCAL_DATA / "data.csv"
        df_merged.to_csv(FILE_PATH)

    else:
        print(f'{LOCAL_DATA} directory is not empty. Please empty it or select'
              ' another destination for LOCAL_DATA if you wish to proceed')

def get_stringency(url_):
    url = url_
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f'Data loader for the {CHALLENGE_NAME} challenge on RAMP.'
    )
    download_from_osf()
