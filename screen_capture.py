import pyautogui
import win32gui
from datetime import datetime  
import time

WINDOW_NAME = 'Speed Dreams 2.3.0-b1-r8589'

def window_screenshot(window_title=WINDOW_NAME):
    """
    Take screenshot of a specefic window based
    on the window title.
    """
    if window_title:
        window = win32gui.FindWindow(None, window_title)
        print(window)
        if window:
            win32gui.SetForegroundWindow(window)
            x, y, x1, y1 = win32gui.GetClientRect(window)
            x, y = win32gui.ClientToScreen(window, (x, y))
            x1, y1 = win32gui.ClientToScreen(window, (x1 - x, y1 - y))
            img = pyautogui.screenshot(region=(x, y, x1, y1))
            return img
        else:
            print('Window not found!')      
    else:
        img = pyautogui.screenshot()
        return img



# while(True):
#     img = window_screenshot()
#     if img:
#         #img.show()
#         filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
#         img.save('./recources/img/'+str(filename)+'.png')
#     time.sleep(1)