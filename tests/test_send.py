from ctypes import *
from multiprocessing import Pool
import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import threading

from src.common.communicator import *

if __name__ =="__main__":
    test_data=np.arange(100000000, dtype=np.int32)
    data_size=(sys.getsizeof(test_data)-96)/1024/1024/1024 # GB
    print("Test data {} GB".format(str(data_size)))

    start= time.time()
    single_process_send(test_data)
    end=time.time()
    print("Single process cost: {} sec; Throuthput {} GBps".format(str(end-start), str(data_size/(end-start))))
    
    print("\n === now testing multiprocessing.Pool ===")
    multi_process_send(10, test_data)

    print("\n === now testing features.ThreadPoolExecutor send ===")
    multi_thread_send_futures(10, test_data)

    print("\n === now testing features.ProcessPoolExecutor send ===")
    multi_process_send_futures_P(10, test_data)

    print("\n === now testing threading send ===")
    multi_thread_send_threading(10, test_data)