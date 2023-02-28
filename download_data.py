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
import numpy as np
LOCAL_DATA = Path(__file__).parent / "data"

CHALLENGE_NAME = "fire"
# you might choosing checking for the correct checksum, if not set
# data_checksum to None
RAMP_FOLDER_CONFIGURATION = {
    "public": dict(
        code="qzrpk",
        archive_name="data.tar.gz",
    ),
}


def get_connection_info():
    "Get connection to OSF and info relative to data."
    osf, folder_name = OSF(), 'public'
    data_config = RAMP_FOLDER_CONFIGURATION[folder_name]
    project = osf.project(data_config['code'])
    store = project.storage('osfstorage')
    return store, data_config

def get_one_element(container):
    elements = [f for f in container]
    assert len(elements) == 1, (
        'There is no element in osf storage'
    )
    return elements[0]


def download_from_osf():
    "Download and uncompress the data from OSF."

    # check if data directory is empty
    if not LOCAL_DATA.exists() or not any(LOCAL_DATA.iterdir()):
        LOCAL_DATA.mkdir(exist_ok=True)

        print("Checking the data URL...", end="", flush=True)
        # Get the connection to OSF
        store, data_config = get_connection_info()


        # Find the file to download from the OSF project
        challenge_file = get_one_element(store.files)
        archive_name = data_config["archive_name"]
        print("Ok.")

        # Download the archive in the data
        ARCHIVE_PATH = LOCAL_DATA / archive_name
        print("Downloading the data...")
        with open(ARCHIVE_PATH, "wb") as f:
            challenge_file.write_to(f)

        # Uncompress the data in the data folder
        print("Extracting now...", end="", flush=True)
        with tarfile.open(ARCHIVE_PATH) as tf:
            tf.extractall(LOCAL_DATA)
        print("Ok.")

        # Clean the directory by removing the archive
        print("Removing the archive...", end="", flush=True)
        ARCHIVE_PATH.unlink()
        print("Ok.")
    else:
        print(
            f"{LOCAL_DATA} directory is not empty. Please empty it or select"
            " another destination for LOCAL_DATA if you wish to proceed"
        )

download_from_osf()