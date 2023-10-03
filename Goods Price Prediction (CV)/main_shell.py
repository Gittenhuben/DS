import numpy as np
from tkinter import *
from tkinter.filedialog import *
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from skimage import transform


SIZE_X = 192
SIZE_Y = 320

WINDOW_X = 640
WINDOW_Y = 480


def load_image(event = None):

    url = askopenfilename(filetypes=[("Pictures","*.jpg *.jpeg *.png *.bmp *.gif"),
                                     ("All Files","*.*")])

    if url:

        info.set("")

        img = Image.open(url)

        img_w, img_h = img.size
        ratio = min(WINDOW_X / img_w, WINDOW_Y / img_h)
        image_resized = img.resize((int(img_w * ratio), int(img_h * ratio)))
        new_image = Image.new('RGB', (WINDOW_X, WINDOW_Y), (240, 240, 240))
        new_image.paste(image_resized,
                     (int((WINDOW_X - int(img_w * ratio)) / 2),
                      int((WINDOW_Y - int(img_h * ratio)) / 2)))

        render = ImageTk.PhotoImage(new_image)
        lab_0.configure(image=render)
        lab_0.image = render

        tk.update()

        img = np.array(img).astype('float32')/255
        img = transform.resize(img, (SIZE_Y, SIZE_X, 3))
        img = np.expand_dims(img, axis=0)

        model = load_model('model9.keras')
        prediction = model.predict(img)
        info.set("Price: " + str(int(round(prediction[0][0]))))

def close_window(event = None):  
        tk.destroy()

tk = Tk()
tk.title("Price Prediction Model")

main_menu = Menu(tk)
tk.config(menu=main_menu)
file_menu = Menu(main_menu)
main_menu.add_cascade(label="MENU", menu=file_menu)
file_menu.add_command(label="Open Image (Ctrl+O)", command=load_image)
file_menu.add_command(label="Exit (Ctrl+Q)", command=close_window)

tk.bind('<Control-o>', load_image)
tk.bind('<Control-q>', close_window)

lab_0=Label(tk,image=None)
lab_0.grid(row=0, column=0)

info = StringVar(tk)
lab_1 = Label(tk, textvariable=info, font=("Arial", 12, "bold"), foreground='black')
lab_1.grid(row=1, column=0)   

tk.geometry("%dx%d" % (WINDOW_X + 4, WINDOW_Y + 30))
tk.mainloop()
