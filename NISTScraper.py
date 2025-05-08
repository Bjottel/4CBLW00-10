#!/usr/bin/env python
# -*- coding: utf-8 -*-
#Credit for the unmodified code and species.txt goes to https://github.com/nj-saquer/IR-Spectra-Prediction-Graph-Models

"""Download all IR spectra available from NIST Chemistry Webbook."""

import os
import re

import requests
from bs4 import BeautifulSoup
from multiprocessing.pool import ThreadPool
import tqdm
import time
import datetime


NIST_URL = 'http://webbook.nist.gov/cgi/cbook.cgi'
EXACT_RE = re.compile('/cgi/cbook.cgi\\?GetInChI=(.*?)$')
ID_RE = re.compile('/cgi/cbook.cgi\\?ID=(.*?)&')
#NOTE: Change these
JDX_PATH = 'jdx'
MOL_PATH = 'mol

MAX_WAIT = 1200

def search_nist_formula(formula, allow_other = False, allow_extra = False, match_isotopes = True, exclude_ions = False, has_ir = True):
    """Search NIST using the specified formula query and return the matching NIST IDs."""
    #print('Searching: %s' % formula)
    params = {'Formula': formula, 'Units': 'SI'}
    if allow_other:
        params['AllowOther'] = 'on'
    if allow_extra:
        params['AllowExtra'] = 'on'
    if match_isotopes:
        params['MatchIso'] = 'on'
    if exclude_ions:
        params['NoIon'] = 'on'
    if has_ir:
        params['cIR'] = 'on'
    response = requests.get(NIST_URL, params=params)
    soup = BeautifulSoup(response.text)
    ids = [re.match(ID_RE, link['href']).group(1) for link in soup('a', href = ID_RE)]
    #print('Result: %s' % ids)
    return ids


def get_jdx(nistid, stype = "IR"):
    print('started %s' % (nistid))
    """Download jdx file for the specified NIST ID, unless already downloaded."""
    filepath_first = os.path.join(JDX_PATH, '%s-%s-0.jdx' % (nistid, stype))

    # A jdx file already exists for this molecule so we don't have to find jdx files again
    if os.path.isfile(filepath_first):
        return False

    # A given molecule may have more than one IR spectrum.
    # To solve this, we keep trying to download spectra from NIST until it doesn't have any spectra anymore.
    index = 0
    waitTime = 60
    while (True):
        jdxResponse = requests.get(NIST_URL, params={'JCAMP': nistid, 'Type': stype, 'Index': index})
        time.sleep(5)

        # Sending too many requests, we have to slow down
        if jdxResponse.text == '##TITLE=Rate limit exceeded.\n##END=\n':
            retry_after = response.headers.get('Retry-After')

            currWait = 0
            # In case NIST provides us with a retry-after, we just have to wait for that amount of time.
            if retry_after:
                try:
                    currWait = int(retry_after)
                except:
                    # If the retry-after is invalid, we just guess the wait time and wait that long.
                    currWait = waitTime
                    waitTime = min(waitTime * 2, MAX_WAIT)
            else:
                # If no retry-after was provided, we just guess the wait time and wait that long.
                currWait = waitTime
                waitTime = min(waitTime * 2, MAX_WAIT)
            print('WAITING %i SECONDS BEFORE NEXT REQUEST')
            time.sleep(currWait)

            continue

        #No more spectra can be found so we can break out of the loop
        if jdxResponse.text == '##TITLE=Spectrum not found.\n##END=\n':
            return False if index == 0 else True

        print('Found a spectrum for nistid %s and index %i' % (nistid, index))
        filepath_new = os.path.join(JDX_PATH, '%s-%s-%i.jdx' % (nistid, stype, index))
        index = index + 1

        with open(filepath_new, 'wb') as file:
            file.write(jdxResponse.content)
        waitTime = 60


def get_mol(nistid):
    """Download mol file for the specified NIST ID, unless already downloaded."""
    filepath = os.path.join(MOL_PATH, '%s.mol' % nistid)

    # A mol file already exists for this molecule so we don't have to look for it again.
    if os.path.isfile(filepath):
        return False

    noResponse = True
    while (noResponse):
        response = requests.get(NIST_URL, params={'Str2File': nistid})
        time.sleep(5)
        if response.text == 'NIST    12121112142D 1   1.00000     0.00000\nCopyright by the U.S. Sec. Commerce on behalf of U.S.A. All rights reserved.\n0  0  0     0  0              1 V2000\nM  END\n':
            # No mol file exists in NIST
            return False

             # Sending too many requests, we have to slow down
        if jdxResponse.text == '##TITLE=Rate limit exceeded.\n##END=\n':
            retry_after = response.headers.get('Retry-After')

            currWait = 0
            # In case NIST provides us with a retry-after, we just have to wait for that amount of time.
            if retry_after:
                try:
                    currWait = int(retry_after)
                except:
                    # If the retry-after is invalid, we just guess the wait time and wait that long.
                    currWait = waitTime
                    waitTime = min(waitTime * 2, MAX_WAIT)
            else:
                # If no retry-after was provided, we just guess the wait time and wait that long.
                currWait = waitTime
                waitTime = min(waitTime * 2, MAX_WAIT)
            print('WAITING %i SECONDS BEFORE NEXT REQUEST')
            time.sleep(currWait)


        print('Found a mol file for nistid %s' % (nistid))
        noResponse = False

    with open(filepath, 'wb') as file:
        file.write(response.content)
        return True

def retreive_data_from_formula(formula):
    ids = search_nist_formula(formula, allow_other = True, exclude_ions = False, has_ir = True)
    for nistid in ids:
        get_mol(nistid)
        get_jdx(nistid)

def get_all_IR():
    """Search NIST for all structures with IR Spectra and download a JDX + Mol file for each."""
    formulae = []
    IDs = []
    with open("species.txt") as data_file:
        entries = data_file.readlines()
        for entry in entries:
            try:
                # The second to last word on any given line is a molecule's formula
                formulae.append(entry.split()[-2])
                # The last word on any given line is a molecule's ID (ie CAS number)
                ID = entry.split()[-1]
                if (ID != "N/A"):
                    IDs.append(ID)
            except:
                IDs.append(entry.strip())

    while True:
        try:
            for nistid in IDs:
                # A lot of molecules don't have any .jdx files, so trying to extract .mol files for these molecules is useless as we can't use their spectra.
                if (get_jdx(nistid)):
                    get_mol(nistid)
        except:
            print("Failed to connect, retrying")
    print("Done!")



if __name__ == '__main__':
    get_all_IR()
