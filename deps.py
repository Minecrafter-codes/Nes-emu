from tkinter import *
from threading import Thread





def install():
    try:
        root = Tk()
        root.geometry('500x500')

        

        my_label = Label(root,
                         text = "Installing dependencies")

        # place the widgets
        # in the gui window
        my_label.pack()
        while True:
            root.update()
    except:
        import traceback; traceback.print_exc()
        print('Warning: Tk is not available right now. gui turned off')

t = Thread(target=install)
t.daemon=True
t.start()
import ensurepip
ensurepip.bootstrap(root=None, upgrade=True, verbosity=1)
from pip._internal import main as pipmain
print('Starting pip install. the deps installer only does this while pip is still supporting this.')
pipmain(['install', '--target=./Python\\Lib\\site-packages', '-r', 'requirements.txt', '--verbose'])

print('done!')

exit()




