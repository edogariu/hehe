from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time
import numpy as np
import pandas as pd
import tqdm
import multiprocessing
import threading
import pickle
import os

PICKLE_FILEPATH = 'data/cite_loci.pkl'

threadLocal = threading.local()
def get_driver():
  driver = getattr(threadLocal, 'driver', None)
  if driver is None:
    chromeOptions = webdriver.ChromeOptions()
    chromeOptions.add_argument("--headless")
    driver = webdriver.Chrome(chrome_options=chromeOptions)
    setattr(threadLocal, 'driver', driver)
  return driver

def init_globals(counter):
    global count
    count = counter

def attempt(subsequence, ret):
        try:
            driver = get_driver()
            driver.get("https://www.ncbi.nlm.nih.gov/genome/gdv/browser/gene/?id=1")
            time.sleep(1.8)
            
            i = 0
            idxs = np.random.permutation(len(subsequence))
            while i < len(subsequence):
                time.sleep(0.1)        
                k = subsequence[idxs[i]]
                
                elem = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, 'loc-search'))).click()
                elem = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, 'loc-search')))
                if i > 0:
                    for _ in range(16):
                        elem.send_keys(Keys.BACKSPACE)
                        time.sleep(0.001)
                elem.send_keys(k)
                elem.send_keys(Keys.ENTER)
                
                elem = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.XPATH, '//div[2]/ul/li/span[2]')))
                t = elem.text
                s = 0
                found = True
                while 'Chr' not in t or (i > 0 and t == ret[subsequence[idxs[i - 1]]]):
                    time.sleep(0.01)
                    elem = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.XPATH, '//div[2]/ul/li/span[2]')))
                    t = elem.text        
                    if s > 100:
                        found = False
                        break
                    s += 1
                if not found: 
                    ret[k] = 'N/A'
                    i += 1
                    continue
                ret[k] = t
                i += 1
            with count.get_lock():
                count.value += 1
        except Exception as e: 
            if 'ENSG' in str(e):
                ret[str(e)] = 'N/A'
            with count.get_lock():
                count.value += 1

def do_thing():
    cite_df = pd.read_hdf('data/train_multi_targets.h5', start=0, stop=1)
    cite_keys = list(cite_df.keys())
    
    L = 17  # length of subsequence for each thread to try and look up
    ncpu = multiprocessing.cpu_count()
    
    counter = multiprocessing.Value('i', 0)  # count of how many processes were finished
    ret = multiprocessing.Manager().dict()

    # handle the already gathered thingies
    if os.path.isfile(PICKLE_FILEPATH):
        with open(PICKLE_FILEPATH, 'rb') as f:
            curr_dict = pickle.load(f)
        curr = list(curr_dict.keys())
        keys = [k for k in cite_keys if k not in curr]
        for k in curr:
            ret[k] = curr_dict[k]
        print(len(keys))

    # assert len(keys) % L == 0, 'should prolly have even jobs'

    inputs = [keys[i: i + L] for i in range(0, len(keys), L)]
    n = len(inputs)
    for i in range(n):
        inputs[i] = (inputs[i], ret)

    print('Spawning {} workers for {} jobs, each of size {}'.format(ncpu, n, L))
    with multiprocessing.Pool(ncpu, initializer=init_globals, initargs=(counter,)) as p:
        p.starmap_async(attempt, inputs[:n])
        prev_counter = counter.value
        with tqdm.tqdm(total=n) as pbar:
            while counter.value < n:
                if counter.value != prev_counter:
                    prev_counter = counter.value
                    pbar.update(1)
                else:
                    time.sleep(0.01)
        p.close()
        p.join()
    
    ret = dict(ret)
    
    with open(PICKLE_FILEPATH, 'wb') as f:
        pickle.dump(ret, f)

if __name__ == '__main__':
    do_thing()
