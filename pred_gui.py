from tkinter import *
import cv2
import screen_capture
import numpy as np
import pred_knn


global lbl_Dispay

IMAGE_HEIGHT = 640
IAMGE_WIDTH = 640
IMAGE_DIR = './recources/img/'


def direction():
  #time.sleep(1)
  img = screen_capture.window_screenshot()
  frame = np.array(img)
  # Convert it from BGR(Blue, Green, Red) to
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  pred = pred_knn.predict_direction(frame)[0]

  if pred == 0:
    lbl_Dispay['text'] = "Left"
  elif pred == 1:
    lbl_Dispay['text'] = "Right"
  elif pred == 2:
    lbl_Dispay['text'] = "Forward"

  window.after(100, direction)

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

direction()
window.mainloop()
