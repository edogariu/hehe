## locate each cite gene
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time
import pandas as pd
import tqdm
import multiprocessing
import threading

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
            while i < len(subsequence):
                time.sleep(0.1)        
                k = subsequence[i]
                
                elem = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, 'loc-search')))
                elem.click()
                if i > 0:
                    for _ in range(16):
                        elem.send_keys(Keys.BACKSPACE)
                        time.sleep(0.001)
                elem.send_keys(k)
                elem.send_keys(Keys.ENTER)
                
                elem = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.XPATH, '//div[2]/ul/li/span[2]')))
                t = elem.text
                s = 0
                while 'Chr' not in t or (i > 0 and t == ret[subsequence[i - 1]]):
                    time.sleep(0.1)
                    elem = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.XPATH, '//div[2]/ul/li/span[2]')))
                    t = elem.text        
                    if s > 150:
                        with count.get_lock():
                            count.value += 1
                        return
                    s += 1
                                
                ret[k] = t
                i += 1
            driver.close()
            with count.get_lock():
                count.value += 1
        except Exception as e: 
            print('err', e)
            driver.close()
            with count.get_lock():
                count.value += 1

def do_thing():
    cite_df = pd.read_hdf('data/train_multi_targets.h5', start=0, stop=1)
    cite_keys = list(cite_df.keys())

    L = 18  # length of subsequence for each thread to try and look up
    ncpu = multiprocessing.cpu_count()
    
    
    counter = multiprocessing.Value('i', 0)  # count of how many processes were finished
    ret = multiprocessing.Manager().dict()

    inputs = [cite_keys[i: i + L] for i in range(0, len(cite_keys), L)]
    n = 2 #len(inputs)
    for i in range(n):
        inputs[i] = (inputs[i], ret)

    print('Spawning {} workers'.format(ncpu))
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
    
    import pickle
    with open('data/cite_loci.pkl', 'wb') as f:
        pickle.dump(ret, f)
    print(list(ret.keys()))

if __name__ == '__main__':
    do_thing()