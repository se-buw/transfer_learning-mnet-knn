from tkinter import *
from datetime import datetime  
import time
import pandas as pd
from pynput import keyboard
from pynput.keyboard import Key
import screen_capture

global lbl_Dispay
IMAGE_HEIGHT = 640
IAMGE_WIDTH = 640
IMAGE_DIR = './recources/img/'

img_list = []
label_list = []

window = Tk()
window.title("Bite The Bytes")
window.geometry("800x75+10+10")
window.resizable(False, False) 
lbl_Dispay = Label(window, 
		 text="",
		 fg = "#10a6eb",
		 font = "Helvetica 30 bold italic")
lbl_Dispay.grid(row=0, column=0, sticky="NW")
lbl_Dispay.place(relx=0.5, rely=0.5, anchor=CENTER)


def on_key_release(key):
    try:
        if key == Key.esc:
            df = pd.DataFrame(list(zip(img_list, label_list))) 
            fn = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
            df.to_csv('./recources/'+fn+'.csv', index=False, header = False)
            window.quit()
        elif key.char == 'd':
            print("Right key clicked")
            lbl_Dispay['text'] = "Right"
            button_click_action('right')
        elif key.char == 'a':
            print("Left key clicked")
            lbl_Dispay['text'] = "Left"
            button_click_action('left')
        elif key.char == 'w':
            print("Up key clicked")
            lbl_Dispay['text'] = "Forward"
            button_click_action('forward')
    except: 
        pass
    

def button_click_action(direction=""):
    img = screen_capture.window_screenshot()
    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    iamge = img.resize((IMAGE_HEIGHT, IAMGE_WIDTH))
    img_list.append(filename+'.jpg')
    label_list.append(direction)
    iamge.save(IMAGE_DIR+str(filename)+'.jpg')
    
listener = keyboard.Listener(
    on_release=on_key_release)
listener.start()

window.mainloop()

