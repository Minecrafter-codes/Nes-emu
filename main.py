#!python3.10
# import everything from tkinter module
try:
    from tkinter import *   
    from tkinter import messagebox
    import tkinter as tk, os, sys
    from tkinter import ttk
    from tkinter import filedialog as fd
    from tkinter.messagebox import showinfo
except:
    from Tkinter import *   
    from Tkinter import messagebox
    import Tkinter as tk, os, sys
    from Tkinter import ttk
    from Tkinter import filedialog as fd
    from Tkinter.messagebox import showinfo
if __name__ == '__main__':
    try:
        print('Starting')
        def reset_controls():
            txt = '''
#\/\/\/\/ this is only to edit
pygame.K_w: ControllerBase.UP,  #<< editing this will destroy the keyboard 
pygame.K_a: ControllerBase.LEFT,
pygame.K_s: ControllerBase.DOWN,
pygame.K_d: ControllerBase.RIGHT,
pygame.K_g: ControllerBase.SELECT,
pygame.K_h: ControllerBase.START,
pygame.K_l: ControllerBase.B,
pygame.K_p: ControllerBase.A,
            '''
            with open('controls.txt', 'w') as f:
                f.write(txt)
                print('Controls Resetted!')
                messagebox.showinfo("Done!", 'Controls Resetted!', parent=root)
        def controls():
            print('Opening Text Editor. Window will freeze while text editor is opened')
            print(f'Os: {sys.platform}')
            print('Note: \nwin32 = Windows \ndarwin = Macos \nlinux = Linux \n')
            if sys.platform == 'win32':
                print('Opening Notepad')
                os.system('notepad controls.txt')
            if sys.platform == 'linux':
                print('Opening Gedit ')
                os.system('gedit ./controls.txt')
            if sys.platform == 'darwin':
                print('Opening Nano ')
                print('WARNING: this emulator is not tested in MacOs. Open controls.txt using any text editor. Default text editor is Nano')
                os.system('nano ./controls.txt')
        def close():
            print('Exiting')
            #root.quit()
            #root.destroy()
            exit()
        def select_files():
            global f
            f = fd.askopenfilename(filetypes=(("Nintendo Entertainment System Files", "*.nes"),("All Files", "*.*"),),)
            my_label.config(text = f"Game: {f}")
        def start():
            while True:
                global f
                if not f:
                    messagebox.showerror("An Error has occurred:", 'No Game has loaded', parent=root)
                    return
                from multiprocessing import Process
                import mainnes as nes
                p = Process(target=nes.nesstart, kwargs={"rom_":f})
                print('Starting Game')
                p.start()
                print('Game started')
                print('=======================================')
                root.iconify()
                root.withdraw()
                p.join()
                root.deiconify()
                print('=======================================')
                print(f'Process Finished with exitcode {p.exitcode}')
                
                if not p.exitcode == 0:
                    try:
                        fp = open('./restart/error.txt', 'r')
                        
                        messagebox.showerror("An Error has occurred:", f'Emulator crashed!\nReason:\n{fp.read()}', parent=root)
                        fp.close()
                        os.remove('./restart/error.txt')
                    except:
                        messagebox.showerror("An Error has occurred:", 'Emulator crashed!\nNo Outputs.', parent=root)
                restart = os.path.isfile('./restart/true.txt')
                del p
                if not restart:
                    return
                else:
                    print('restarting emulator')
                    os.remove("./restart/true.txt")
        # create a tkinter window
        root = Tk()             
        
        # Open window having dimension 100x100
        root.geometry('500x500')
        print('Window Created')
         
        # Create a Button
        btn4 = Button(root, text = 'Reset Controls', bd = '5',
                                  command = reset_controls)
        btn3 = Button(root, text = 'Controls', bd = '5',
                                  command = controls)
        btn2 = Button(root, text = 'Load Game!', bd = '5',
                                  command = select_files)
        btn = Button(root, text = 'Start!', bd = '5',
                                  command = start)
        print('Button widget ready')
        # create a Label widget
        my_label = Label(root,
                         text = "Game: No game")
         
        # place the widgets
        # in the gui window
        my_label.pack()
        print('Label widget Ready')


        # Set the position of button on the top of window.  
        btn2.pack(side = 'top')
        #btn4.pack(side = 'top')
        #btn3.pack(side = 'top')
        btn.pack(side = 'top')   
        print('Widgets created')
        
        
        
        try:
            root.protocol("WM_DELETE_WINDOW", root.destroy)
            
            #root.update_idletasks()
            print('Starting Mainloop')
            root.mainloop()
        except KeyboardInterrupt:
            raise
        except:
            pass
        close()
            

    except KeyboardInterrupt:
        import traceback as tb, sys
        sys.stderr.write(f'FATAL ERROR: \n{tb.format_exc()} \n')
        try:
            input('Press enter to exit ')
        except (KeyboardInterrupt, EOFError, ValueError):
            sys.stderr.write('^C')
            import os
            os._exit(1  )
    except Exception as e:
        import traceback as tb
        messagebox.showerror(f"An Error has occurred: {e}", f'Launcher Crashed: \n{tb.format_exc()}')
        for i in range(10000000):
            pass