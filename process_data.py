from utilities.data import *
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config_file', type=str, default='metagenomics_and_metabolomics_example.cfg')
    args = parser.parse_args()

    pData = ProcessData(args.config_file)