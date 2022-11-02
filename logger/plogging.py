from multiprocessing import *
import Embedded.futures, sys, atexit
from datetime import datetime

def shutdown():
    #print('plogger shutdown')
    global P, P2, P3, P4 ,P5, executor
    #from logger import plogging
    try: 
         P.cancel()
    except: pass
    try: 
         P2.cancel()
    except: pass
    try: 
         P3.cancel()
    except: pass
    try: 
         P4.cancel()
    except: pass
    try: 
         P5.cancel()
    except: pass
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except: pass
    
    

executor = Embedded.futures.ProcessPoolExecutor(max_workers=50) 
atexit.register(shutdown)
def trace(msg):
    pass
    #P = executor.submit(print, f'[{datetime.now().time()}], Debug   : {msg}', flush=True)
def debug(msg):
    P = executor.submit(print, f'[{datetime.now().time()}], Debug   : {msg}', flush=True)
def info(msg):               
    P2 = executor.submit(print, f'[{datetime.now().time()}], Info    : {msg}', flush=True)
def warning(msg):               
    P3 = executor.submit(print, f'[{datetime.now().time()}], Warning : {msg}', flush=True)
def error(msg):              
    P4 = executor.submit(print, f'[{datetime.now().time()}], Error   : {msg}', flush=True, file=sys.stderr)
def fatal(msg):              
    P5 = executor.submit(print, f'[{datetime.now().time()}], Fatal   : {msg}', flush=True, file=sys.stderr)