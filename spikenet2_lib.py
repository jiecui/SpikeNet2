# Library for Spikenet2

# 2025 Richard J. Cui. Created: Fri 09/12/2025 04:16:14.055411 PM
# $Revision: 0.3 $  $Date: Thu 10/02/2025 02:28:14.352718 PM $
#
# Mayo Clinic Foundation
# Rochester, MN 55901, USA
#
# Email: Cui.Jie@mayo.edu

# Imports
import os
import shutil
import socket


# Define functions
def copy_new_files(source_dir, dest_dir):
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Get list of files in source directory
    source_files = [
        f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))
    ]

    # Get list of files in destination directory
    dest_files = [
        f for f in os.listdir(dest_dir) if os.path.isfile(os.path.join(dest_dir, f))
    ]

    # Find new files (files in source but not in destination)
    new_files = [f for f in source_files if f not in dest_files]

    if new_files:
        print(f"Copying {len(new_files)} new files.")
        for filename in new_files:
            source_file = os.path.join(source_dir, filename)
            dest_file = os.path.join(dest_dir, filename)

            try:
                shutil.copy2(source_file, dest_file)
            except Exception as e:
                print(f"  Error copying {filename}: {str(e)}")
    else:
        print("No new files to copy.")


def get_host_name():
    return socket.gethostname()


def get_proj_root():
    return os.path.dirname(os.path.abspath(__file__))


def get_database_root():
    host_name = get_host_name()
    if host_name == "R5504550":  # lab main desktop
        return os.path.join(
            "H:\\", "Documents", "Richard", "Datasets", "spikenet2_datasets"
        )
    elif host_name == "aif-richard-us-central1-a":  # MCC workbench instance
        return os.path.join(
            "/home",
            "ext_cui_jie_mayo_edu",
            "Documents",
            "Richard",
            "Datasets",
            "spikenet2_datasets",
        )
    elif host_name == "bnel-lambda1" or host_name == "bnel-lambda2":  # MSEL lab servers
        return os.path.join(
            "/mnt", "eplab", "Personal", "Richard", "Datasets", "spikenet2_datasets"
        )
    else:
        raise ValueError(f"Unknown host name {host_name} for database root.")


def get_output_root():
    host_name = get_host_name()
    if (
        host_name == "R5504550" or host_name == "aif-richard-us-central1-a"
    ):  # lab main desktop and MCC workbench instance
        return get_database_root()
    elif host_name == "bnel-lambda1" or host_name == "bnel-lambda2":  # MSEL lab servers
        return os.path.join(
            "/mnt",
            "Hydrogen",
            "richard",
            "Documents",
            "Richard",
            "Datasets",
            "spikenet2_datasets",
        )
    else:
        raise ValueError(f"Unknown host name {host_name} for output root.")


# [EOF]
