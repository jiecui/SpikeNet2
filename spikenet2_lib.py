# Library for Spikenet2

# 2025 Richard J. Cui. Created: Fri 09/12/2025 04:16:14.055411 PM
# $Revision: 0.1 $  $Date: Fri 09/12/2025 04:16:14.055411 PM $
#
# Mayo Clinic Foundation
# Rochester, MN 55901, USA
#
# Email: Cui.Jie@mayo.edu

# Imports
import os
import socket


# Define functions
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
            "FuseMount",
            "richard",
            "Documents",
            "Richard",
            "Documents",
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
