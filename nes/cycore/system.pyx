# import pyximport; pyximport.install()

from .mos6502 cimport MOS6502
from .memory cimport NESMappedRAM
from .ppu cimport NESPPU
from .apu cimport NESAPU, SAMPLE_RATE

from nes.rom import ROM
from nes.peripherals import Screen, ScreenGL, KeyboardController, ControllerBase
from nes.utils import load_palette

import logging
import time
import warnings
import tracemalloc
tracemalloc.start()
from tkinter import *


#root = Tk()
#root.geometry('500x500')
         
        
# create a Label widget

#root.update()

# unfortunately, pygame's audio handling does not do what we need with gapless playback of short samples
# we therefore use pyaudio, but it would be nice to make it optional
try:
    import pyaudio
    has_audio = True
except ImportError:
    has_audio = False

# would like to make this not depend on pygame
try:
    import pygame
    has_pygame = True
except ImportError:
    has_pygame = False

# would like to know if PyOpenGL is available to avoid using OpenGL screen if not
try:
    import OpenGL.GL
    has_opengl = True
except ImportError:
    has_opengl = False

# only use numpy for return of frame during headless operation, so don't depend on this in general
try:
    import numpy as np
    has_numpy = True
except ImportError:
    has_numpy = False
try: import keyboard
except: pass


# custom log levels to control the amount of logging (these are mostly unused because logging is mostly disabled)
LOG_MEMORY = 5
LOG_PPU = 6
LOG_CPU = 7


LOG_MEMORY = True
LOG_PPU  = True
LOG_CPU = True


cdef class InterruptListener:
    def __init__(self):
        self._nmi = False
        self._irq = False   # the actual IRQ line on the 6502 rests high and is low-triggered
        self.dma_pause = False
        self.dma_pause_count = 0

    cdef void raise_nmi(self):
        self._nmi = True

    cdef void reset_nmi(self):
        self._nmi = False

    cdef void raise_irq(self):
        self._irq = True

    cdef void reset_irq(self):
        self._irq = False

    cdef void reset_dma_pause(self):
        self.dma_pause = 0
        self.dma_pause_count = 0

    cdef void raise_dma_pause(self, int type):
        self.dma_pause = type
        self.dma_pause_count += 1  # can (and this will be rare) get multiple DMC DMA pauses during OAM DMA

    cdef int any_active(self):
        return self._nmi or self._irq or self.dma_pause

    cdef int nmi_active(self):
        return self._nmi

    cdef int irq_active(self):
        return self._irq
## Reset keyboard shortcut
def done():
    fp = open('./restart/true.txt', 'w')
    fp.close()
    import os, sys
    pid = os.getpid()
    print(f'Restarting. Pid: {pid}. OS: {sys.platform}')
    if sys.platform.startswith('freebsd'):
        os.system(f'kill -15 {pid}')
    elif sys.platform.startswith('linux'):
        os.system(f'kill -15 {pid}')
    elif sys.platform.startswith('aix'):
        os.system(f'kill -15 {pid}')
    elif sys.platform.startswith('darwin'):
        os.system(f'kill -15 {pid}')
    elif sys.platform.startswith('win32'):
        os.system(f'taskkill /pid {pid}')
    elif sys.platform.startswith('cygwin'):
        print('Warning: using terminal emulatores like MinGW, Git, and Cygwin can cause problems')
        os.system(f'taskkill /pid {pid}')
    else:
        
        print(f'Warning: Unknown os: {sys.platform}')
        
        print('Aborting. ')
        os.abort()
    return

cdef class NES:
    """
    The NES system itself, combining all of the parts, and the interlinks
    """
    OSD_Y = 3
    OSD_FPS_X = 5
    OSD_VOL_X = 110
    OSD_MUTE_X = 200
    OSD_NOTE_X = 100

    OSD_TEXT_COLOR = (0, 255, 0)
    OSD_NOTE_COLOR = (255, 128, 0)
    OSD_WARN_COLOR = (255, 0, 0)

    STATS_CALC_PERIOD_S = 1.0        # how often to calculate the adaptive audio rate and fps counter
    
    def __init__(self,
                 rom_file,                  # the rom file to load
                 screen_scale=3,            # factor by which to scale the screen
                 log_file=None,             # file to log to (logging is largely turned off by default)
                 log_level=None,            # level of logging (logging is largely turned off by default)
                 opengl=False,              # use opengl for screen rendering
                 sync_mode=SYNC_AUDIO,      # audio / video sync mode
                 verbose=True,              # whether to print out cartridge info at startup
                 show_nametables=False,     # shows the nametables alongside the main screen (for debug, not opengl)
                 vertical_overscan=False,   # show the top and bottom 8 pixels (not usually visible on CRT TVs)
                 horizontal_overscan=False, # show the left and right 8 pixels (often not visible on CRT TVs)
                 palette_file=None,         # supply a palette file to use; None gives default
                 headless=False,            # runs the nes in headless mode without the pygame screen being started
                 ):
        """
        Build a NES and cartridge from the bits and pieces we have lying around plus some rom data!  Also do some things
        like set up logging, make sure the options are compatible (and warn if not) etc.
        """
        global headles, openg
        headles = headless
        openg = opengl
        if sync_mode == SYNC_AUDIO and not has_audio:
            self.sync_mode = SYNC_PYGAME
            warnings.warn("Selected sync_mode is SYNC_AUDIO, but audio is not available, probably because pyaudio is "
                          "not available on the system")
        else:
            self.sync_mode = sync_mode

        if opengl and not has_opengl:
            warnings.warn("OpenGL is not available, possibly because PyOpenGL is not installed (try 'pip install "
                          "PyOpenGL' or start with option opengl=False).  Starting with PyGame (SDL) based screen.")

        if opengl and show_nametables:
            warnings.warn("Nametable display not supported in OpenGL mode.  Use opengl=False to display.")

        # record overscan preference
        self.h_overscan = horizontal_overscan
        self.v_overscan = vertical_overscan

        # set up the logger
        self.init_logging(log_file, log_level)

        # the interrupt listener here is not a NES hardware device, it is an interrupt handler that is used to pass
        # interrupts between the PPU, APU and sometimes even cartridge and CPU in this emulator
        self.interrupt_listener = InterruptListener()

        rom = ROM(rom_file, verbose=verbose)
        # the cartridge is a piece of hardware (unlike the ROM, which is just data) and must come first because it
        # supplies bits of hardware (memory in most cases, both ROM and RAM) all over the system, so it
        # affects the actual hardware configuration of the system.
        self.cart = rom.get_cart(self.interrupt_listener)

        # game controllers have no dependencies
        if has_pygame:
            self.controller1 = KeyboardController()
        else:
            self.controller1 = ControllerBase(active=True)
        self.controller2 = KeyboardController()   # connect a second gamepad, but make it inactive for now

        # load the requested palette
        if palette_file is not None:
            try:
                
                palette = load_palette(palette_file)
                #raise Exception
            except: 
                    logger.WARNING('Palette is not available. Setting to none')
                    palette = None
        else:
            palette = None

        # set up the APU and the PPU
        self.ppu = NESPPU(cart=self.cart, interrupt_listener=self.interrupt_listener, palette=palette)
        self.apu = NESAPU(interrupt_listener=self.interrupt_listener)

        # screen needs to have the PPU
        if has_pygame and not headless:
            if opengl and has_opengl:
                self.screen = ScreenGL(ppu=self.ppu,
                                       scale=screen_scale,
                                       vsync=True if sync_mode == SYNC_VSYNC else False,
                                       vertical_overscan=vertical_overscan,
                                       horizontal_overscan=horizontal_overscan
                                      )
            else:
                self.screen = Screen(ppu=self.ppu,
                                     scale=screen_scale,
                                     vsync=True if sync_mode == SYNC_VSYNC else False,
                                     nametable_panel=show_nametables,
                                     vertical_overscan=vertical_overscan,
                                     horizontal_overscan=horizontal_overscan
                                    )
        else:
            self.screen=None

        self.screen_scale = screen_scale

        # due to memory mapping, lots of things are connected to the main memory
        self.memory = NESMappedRAM(ppu=self.ppu,
                                   apu=self.apu,
                                   cart=self.cart,
                                   controller1=self.controller1,
                                   controller2=self.controller2,
                                   interrupt_listener=self.interrupt_listener
                                   )

        # one nasty wrinkle here is that we have to give the apu's dmc channel access to the memory:
        self.apu.dmc.memory = self.memory

        # only the memory is connected to the cpu, all access to other devices is done through memory mapping
        self.cpu = MOS6502(memory=self.memory,
                           undocumented_support_level=2,  # a few NES games make use of some more common undocumented instructions
                           stack_underflow_causes_exception=True,
                           ignore_halt=False)
        
            
        '''try:
            
            keyboard.add_hotkey("ctrl + c", self.cpu.reset)
        except Exception as e:
            print(e)
            logger.debug('Ctrl + c wont work')
        try:
            
            keyboard.add_hotkey("ctrl + z", done)
        except Exception as e:
            print(e)
            logger.debug('Ctrl + z wont work')'''
        # Let's get started!  Reset the cpu so we are ready to go...
        self.cpu.reset()

    def init_logging(self, log_file, log_level):
        """
        Initialize the logging; set the log file and the logging level (LOG_MEMORY, LOG_PPU, LOG_CPU are all below
        logging.DEBUG)
        """
        #if log_file is None or log_level is None:
        if log_level is None:
            logging.disable(level=logging.CRITICAL)
            return

        #logging.addLevelName(LOG_MEMORY, "MEMORY")  # set a low level for memory logging because it is so intense
        #logging.addLevelName(LOG_PPU, "PPU")
        #logging.addLevelName(LOG_CPU, "CPU")
        #logging.addLevelName(1, "MEMORY")  # set a low level for memory logging because it is so intense
        #logging.addLevelName(1, "PPU")
        #logging.addLevelName(2, "CPU")
        # set up logging to the format as we required
        # format='%(asctime)-15s %(processName)s %(threadName)s %(source)-5s %(message)s'
        
        #logging.basicConfig(#level=logging.WARNING,#logging.WARNING
        #                    format='%(asctime)-1s %(processName)s %(threadName)s %(source)-5s %(message)s'
        #                    )
        #logging.root.setLevel(logging.WARNING #log_level
        #                      )
        #logging.getLogger("exchangelib.folders").disabled = True
        #logging.getLogger("exchangelib.services").disabled = True
        global logger
        logFormatter = logging.Formatter(fmt='%(asctime)s ||  %(name)s | %(levelname)s : %(message)s')

        # create logger
        logger = logging.getLogger('SYSTEM')
        logger.setLevel(logging.DEBUG)

        # create console handler
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.DEBUG)
        consoleHandler.setFormatter(logFormatter)

        # Add console handler to plogger
        logger.addHandler(consoleHandler)

    cdef int step(self, int log_cpu):
        """
        The heartbeat of the system.  Run one instruction on the CPU and the corresponding amount of cycles on the
        PPU (three per CPU cycle, at least on NTSC systems).
        """
        cdef int cpu_cycles=0
        cdef bint vblank_started=False

        if self.interrupt_listener.any_active():
            if self.interrupt_listener.nmi_active():
                cpu_cycles = self.cpu.trigger_nmi()  # raises an NMI on the CPU, but this does take some CPU cycles
                # should we do this here or leave this up to the triggerer?  It's really up to the triggerer to put the NMI
                # line into its "off" position, but if the line remains in the same state the CPU will not trigger another
                # NMI anyway.  Electronics lend themselves to edge-triggered events, whereas software is more naturally
                # pulse triggered in some ways (a thing happens then returns).
                self.interrupt_listener.reset_nmi()
                if log_cpu:
                    logging.log(logging.INFO, "NMI Triggered", extra={"source": "Interrupt"})
            elif self.interrupt_listener.irq_active():
                # note, cpu_cycles can be zero here if the cpu is set to ignore irq
                cpu_cycles = self.cpu.trigger_irq()
                if cpu_cycles > 0:
                    # the cpu reacted to the interrupt, so clear it here so that it won't re-trigger the CPU;
                    # if the cpu does not act on this interrupt, it will remain here (unless cleared by the caller) and
                    # will try to trigger again on the next cycle
                    self.interrupt_listener.reset_irq()
                    if log_cpu:
                        logger.debug('IRQ Triggered')
                        #logging.log(logging.INFO, "IRQ Triggered", extra={"source": "Interrupt"})
            elif self.interrupt_listener.dma_pause:
                # https://wiki.nesdev.com/w/index.php/PPU_OAM#DMA
                cpu_cycles = self.cpu.dma_pause(self.interrupt_listener.dma_pause,
                                                self.interrupt_listener.dma_pause_count
                                                )
                if log_cpu:
                    #logging.log(logging.INFO,
                    #            "OAM DMA (type: {}, count: {})".format(self.interrupt_listener.dma_pause,
                    #                                                   self.interrupt_listener.dma_pause_count),
                    #            extra={"source": "Interrupt"}
                    #            )
                    logger.debug("OAM DMA (type: {}, count: {})".format(self.interrupt_listener.dma_pause,self.interrupt_listener.dma_pause_count))

                self.interrupt_listener.reset_dma_pause()
                had_dma_pause = True
        if cpu_cycles == 0:
            cpu_cycles = self.cpu.run_next_instr()
            if log_cpu:
                logger.debug(self.cpu.log_line(self.ppu.line, self.ppu.pixel))
                #logging.log(logging.INFO, self.cpu.log_line(self.ppu.line, self.ppu.pixel) , extra={"source": "CPU"})

        vblank_started = self.ppu.run_cycles(cpu_cycles * PPU_CYCLES_PER_CPU_CYCLE)
        self.apu.run_cycles(cpu_cycles)
        if had_dma_pause and self.interrupt_listener.dma_pause:
            # any apu dmc dma pause that occurred during an oam dma pause should average 2 rather than 4 cycles
            self.interrupt_listener.dma_pause = DMC_DMA_DURING_OAM_DMA
        return vblank_started

    cpdef run(self):
        """
        Run the NES indefinitely (or until some quit signal); this will only exit on quit.
        There is some PyGame specific stuff in here in order to handle frame timing and checking for exits
        """
        cdef int vblank_started=False
        cdef float volume = 0.5
        cdef double fps, t_start=0., dt=-0.
        cdef bint show_hud=True, log_cpu=False, mute=False, audio_drop=False
        cdef int frame=0, frame_start=0, cpu_cycles=0, adaptive_rate=0, buffer_surplus=0
        cdef bint audio=has_audio
        
        if not has_pygame:
            raise RuntimeError("Cannot run() without pygame; only headless operation is supported.")

        if audio:
            p = pyaudio.PyAudio()
        else:
            warnings.warn("Audio is unavailable, probably because pyaudio is not installed.")

        pygame.init()
        clock = pygame.time.Clock()

        # this has to come after pygame init for some reason, or pygame won't start :(
        if audio:
            try:
                player = p.open(format=pyaudio.paInt16,
                                channels=1,
                                rate=SAMPLE_RATE,
                                output=True,
                                frames_per_buffer=AUDIO_CHUNK_SAMPLES,  # 400 is a half-frame at 60Hz, 48kHz sound
                                stream_callback=self.apu.pyaudio_callback,
                                )
                player.start_stream()
            except Exception as e:
                audio = False
                logger.warning(f'Audio: {e}')
        t_start = time.time()
        
        '''cpulabel = Label(root, text = f"")
        my_label = Label(root,
             text = f"Memory: ")
        my_label2 = Label(root,
                         text = f"Peak: ")
        l3 = Label(root, text = f"Fps: ")
        l4 = Label(root, text = f"frame: ")
        l5 = Label(root, text = f"Events: ")
        my_label.pack()
        my_label2.pack()
        l3.pack()
        l4.pack()
        l5.pack()'''
        
        while True:
            current, peak = tracemalloc.get_traced_memory()    
            #my_label.config(text = f"Memory: {current/1024*1024}Mb")  
            #my_label2.config(text = f"Peak: {peak/1024*1024}Mb")
            #l3.config(text = f"Fps: {fps}")
            #l4.config(text = f"frame: {frame}")
            #l5.config(text = f"Events: {pygame.event.get()}")
            vblank_started=False
            while not vblank_started:
                vblank_started = self.step(log_cpu)

            # update the controllers once per frame
            self.controller1.update()
            self.controller2.update()

            cpu_cycles = self.cpu.cycles_since_reset

            if time.time() - t_start > self.STATS_CALC_PERIOD_S:
                # calcualte the fps and adaptive audio rate (only used when non-audio sync is in use)
                dt = time.time() - t_start
                fps = (frame - frame_start) * 1.0 / dt
                buffer_surplus = self.apu.buffer_remaining() - TARGET_AUDIO_BUFFER_SAMPLES
                
                # adjust the rate based on the buffer change
                adaptive_rate = int(SAMPLE_RATE - 0.5 * buffer_surplus / dt)
                adaptive_rate = max(SAMPLE_RATE - MAX_RATE_DELTA, min(adaptive_rate, SAMPLE_RATE + MAX_RATE_DELTA))
                t_start = time.time()
                frame = 0
                frame_start = frame
                self.cpu.cycles_since_reset = 0
                
            if show_hud:
                # display information on the HUD (turn on/off with '1' key)
                self.screen.add_text("{:.0f} fps, {}Hz, {} samples".format(fps, self.apu.get_rate(), self.apu.buffer_remaining()),
                                     (self.OSD_FPS_X, self.OSD_Y),
                                     self.OSD_TEXT_COLOR if fps > TARGET_FPS - 3 else self.OSD_WARN_COLOR)
                if log_cpu:
                    self.screen.add_text("logging cpu", (self.OSD_NOTE_X, self.OSD_Y), self.OSD_NOTE_COLOR)
                if mute:
                    self.screen.add_text("MUTE", (self.OSD_MUTE_X, self.OSD_Y), self.OSD_TEXT_COLOR)
            
            # Check for an exit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print('Shutting down')
                    pygame.display.set_caption('Shutting down')
                    print('Pygame Exiting')
                    pygame.quit()
                    print('Pygame exited, Stopping Tracemalloc')
                    #root.destroy()
                    tracemalloc.stop()
                    print('Tracemalloc stopped')
                    if audio:
                        print('Stopping Audio')
                        player.stop_stream()
                        player.close()
                        p.terminate()
                        print('Done')
                    import logger as loging
                    import logger.plogging as ploging
                    loging.shutdown()
                    ploging.shutdown()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_BACKSPACE:
                        print('Terminating Python')
                        import ctypes
                        p = ctypes.pointer(ctypes.c_char.from_address(5))
                        p[0] = b'x'
                    if event.key == pygame.K_1:
                        show_hud = not show_hud
                    if event.key == pygame.K_0:
                        if not mute:
                            self.apu.set_volume(0)
                            mute = True
                        else:
                            self.apu.set_volume(volume)
                            mute = False
                    if event.key == pygame.K_MINUS:
                        #volume = max(0, volume - 0.1)
                        volume = volume - 0.1
                        self.apu.set_volume(volume)
                        self.screen.add_text("volume: " + "|" * int(10 * volume),
                                             (self.OSD_VOL_X, self.OSD_Y),
                                             self.OSD_TEXT_COLOR, ttl=30)
                        mute=False
                    if event.key == pygame.K_EQUALS:
                        #volume = min(1, volume + 0.1)
                        volume = volume + 0.1
                        self.apu.set_volume(volume)
                        self.screen.add_text("volume: " + "|" * int(10 * volume),
                                             (self.OSD_VOL_X, self.OSD_Y),
                                             self.OSD_TEXT_COLOR, ttl=30)
                        mute=False
                    if event.key == pygame.K_2:
                        log_cpu = not log_cpu
                    if event.key == pygame.K_ESCAPE:
                        print('Shutting down')
                        pygame.display.set_caption('Shutting down')
                        print('Pygame Exiting')
                        pygame.quit()
                        print('Pygame exited, Stopping Tracemalloc')
                        #root.destroy()
                        tracemalloc.stop()
                        print('Tracemalloc stopped')
                        if audio:
                            print('Stopping Audio')
                            player.stop_stream()
                            player.close()
                            p.terminate()
                            print('Done')
                        import logger as loging
                        import logger.plogging as ploging
                        loging.shutdown()
                        ploging.shutdown()
                        print('Returning')
                        return
                    if event.key == pygame.K_DELETE:
                        done()
                    if event.key == pygame.K_HOME:
                        self.cpu.reset()
                        

            # show the display (if using SYNC_VSYNC mode, this should provide a sync, which must be at 60Hz)
            self.screen.show()
            
            if self.sync_mode == SYNC_AUDIO:
                # wait for the audio buffer to empty, but only if the audio is playing
                while self.apu.buffer_remaining() > TARGET_AUDIO_BUFFER_SAMPLES and audio and player.is_active():
                    clock.tick(500)  # wait for about 2ms (~= 96 samples)
            elif self.sync_mode == SYNC_VSYNC or self.sync_mode == SYNC_PYGAME:
                # here we rely on an external sync source, but allow the audio to adapt to it
                if frame > 2 * TARGET_FPS:  # wait a bit before doing this since startup can be slow
                    self.apu.set_rate(adaptive_rate)
            else:
                # no sync at all, go as fast as we can!
                pass

            if self.sync_mode == SYNC_PYGAME:
                # if we are using pygame sync, we have to supply our own clock tick here
                clock.tick(TARGET_FPS)
            try:
                if (audio
                    and not player.is_active()
                    and self.apu.buffer_remaining() > TARGET_AUDIO_BUFFER_SAMPLES
                    and (frame % 10 == 0)
                    ):
                    logger.warning('Audio stream stopped. Restarting audio')
                    # sometimes the audio stream stops (e.g. if it runs out of samples) and has to be restarted or else the
                    # sound will stop.  Here try to (re)start the stream if it is not running if there is audio waiting.
                    audio_drop=True
                    try:
                        player.stop_stream()
                        player.close()
                        player = p.open(format=pyaudio.paInt16,
                                        channels=1,
                                        rate=SAMPLE_RATE,
                                        output=True,
                                        frames_per_buffer=AUDIO_CHUNK_SAMPLES,  # 400 is a half-frame at 60Hz, 48kHz sound
                                        stream_callback=self.apu.pyaudio_callback,
                                        
                                        )
                        player.start_stream()
                    except Exception as e:
                        logger.warning(f'Audio: {e}')
                    #root.update()
                else:
                    audio_drop = False
            except:
                pass
            frame += 1
            
                         
            
            #pygame.display.set_caption(f'Fps: {fps}, frame: {frame}, Hz: {self.apu.get_rate()}, audio: {audio} "Audio Drop={audio_drop}", Events: {pygame.event.get()}, cpu cycles: {cpu_cycles}, vblank_started {vblank_started}, Samples: {self.apu.buffer_remaining()},Adaptive Rate: {adaptive_rate}, Buffer Surplus: {buffer_surplus}, audio chunk samples: {AUDIO_CHUNK_SAMPLES}, Volume: {volume}, Mem_usage: {current}, Peak Mem_usage: {peak}')
            
            pygame.display.set_caption(f'Fps: {round(fps)}, frame: {frame}, Hz: {self.apu.get_rate()}, audio: {audio} "Audio Drop={audio_drop}", cpu cycles: {cpu_cycles}, vblank_started {vblank_started}, Buffer Surplus: {buffer_surplus}, audio chunk samples: {AUDIO_CHUNK_SAMPLES}, Volume: {round(volume)}, Mem_usage: {current/(1024*1024)}MB, Peak Mem_usage: {round(peak/(1024*1024))}MB, Audio_cpu_usage: {player.get_cpu_load() if audio else None}%')

    cpdef object run_frame_headless(self, int run_frames=1, object controller1_state=[False] * 8, object controller2_state=[False] * 8):
        """
        Runs a single frame of the emulator and returns an array containing the screen output.
        :param run_frames: number of frames to run; controller will be in same state for all frames
        :param controller1_state: controller1 state as an array in order A, B, select, start, up, down, left, right
        :param controller2_state: controller2 state as an array in order A, B, select, start, up, down, left, right
        :return:
        """
        cdef int w, h, frame
        cdef bint vblank_started

        h = 240 if self.v_overscan else 224
        w = 256 if self.h_overscan else 240
        buffer = np.zeros((w, h), dtype=np.uint32)
        cdef unsigned int [:, :] buffer_mv = buffer

        # set the controller state (same for all frames)
        self.controller1.set_state(controller1_state)
        self.controller2.set_state(controller2_state)

        frame = 0
        while frame < run_frames:
            vblank_started=False
            while not vblank_started:
                vblank_started = self.step(log_cpu=False)
            frame += 1

        self.ppu.copy_screen_buffer_to(buffer_mv, v_overscan=self.v_overscan, h_overscan=self.h_overscan)

        # converts to an RGB buffer with one channel per color
        buffer_rgb = buffer.view(dtype=np.uint8).reshape((w, h, 4))[:, :, np.array([2, 1, 0])].swapaxes(0, 1)
        return buffer_rgb







