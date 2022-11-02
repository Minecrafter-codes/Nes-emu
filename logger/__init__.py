import Embedded.futures, sys, atexit

from datetime import datetime


def shutdown():
    global _, _2, _3, _4 ,_5, executor
    try: _.cancel()
    except: pass
    try: _2.cancel()
    except: pass
    try: _3.cancel()
    except: pass
    try: _4.cancel()
    except: pass
    try: _5.cancel()
    except: pass
    try:
        executor.shutdown(wait=False, cancel_futures=True)
        del executor
    except:
        pass
executor = Embedded.futures.ThreadPoolExecutor(max_workers=5) #at least 100 workers in case of lag
atexit.register(shutdown)
def debug(msg):
    _ = executor.submit(print,f'[{datetime.now().time()}], Debug   : {msg}', flush=True)
def info(msg):               
    _2 = executor.submit(print,f'[{datetime.now().time()}], Info    : {msg}', flush=True)
def warning(msg):               
    _3 = executor.submit(print,f'[{datetime.now().time()}], Warning : {msg}', flush=True)
def error(msg):              
    _4 = executor.submit(print,f'[{datetime.now().time()}], Error   : {msg}', flush=True, file=sys.stderr)
def fatal(msg):              
    _5 = executor.submit(print, f'[{datetime.now().time()}], Fatal   : {msg}', flush=True, file=sys.stderr)