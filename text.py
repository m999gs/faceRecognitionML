import keras
import cv2
import os, csv
import numpy as np
from keras.models import Sequential
from keras.optimizers import adam

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
import PIL.Image
import tkinter as tr
from tkinter import *
import tkinter.font as font
import tkinter.ttk as ttk

import sklearn
from sklearn.metrics import accuracy_score


face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml")
root_dir = os.path.dirname(os.path.abspath(__file__))


def TrainModel():
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    temp = root_dir + "/trainImages/"
    path_array = [os.path.join(temp, i) for i in os.listdir(temp)]
    for img in path_array:
        if img.endswith(".png"):
            PILImage = PIL.Image.open(img).convert("L")
            image_array = np.array(PILImage, "uint8") / 255
            x_train.append(image_array)
            # Adding the name of Candidate in y_train as string.
            extractName = int(img.split("#")[2])
            y_train.append(extractName)
    temp = root_dir + "/TestingImages/"
    path_array = [os.path.join(temp, i) for i in os.listdir(temp)]
    for img in path_array:
        if img.endswith(".png"):
            PILImage = PIL.Image.open(img).convert("L")
            image_array = np.array(PILImage, "uint8") / 255
            x_test.append(image_array)
            extractName = img.split("#")[1]
            y_test.append(extractName)



    # cnn_model = Sequential()
    # cnn_model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape = (28, 28, 1)))
    # cnn_model.add(Conv2D(32, kernel_size=3, activation='relu'))
    # cnn_model.add(Flatten())
    # cnn_model.add(Dense(10, activation='softmax'))
    #
    # cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # cnn_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)
    #
    # cnn_model.predict(x_test)



def faceDetect():
    name = ent.get().replace(" ", "")
    roll = ent2.get().replace(" ", "")
    if name.isalpha() and roll.isnumeric():
        cap = cv2.VideoCapture(0)
        train_image_folder_path = root_dir + "/trainImages/"
        student_details_path = root_dir + "/RegisteredStudents/"
        testing_data = root_dir + "/TestingImages/"
        if not os.path.exists('trainImages'):
            os.makedirs('trainImages')
        if not os.path.exists('RegisteredStudents'):
            os.makedirs('RegisteredStudents')
        if not os.path.exists('TestingImages'):
            os.makedirs('TestingImages')
        i = 0

        while (True):
            ret, frame = cap.read()
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=8)

            for x, y, w, h in faces:
                roi_gray = gray_img[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                img_item = "#" + name + "#" + roll + "#" + str(i) + ".png"
                i += 1
                cv2.imwrite(train_image_folder_path + img_item, roi_gray)
                if i<=20:
                    cv2.imwrite(testing_data + img_item, roi_gray)
                color = (0, 255, 0)
                stroke = 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)
            cv2.imshow('frame', frame)
            if cv2.waitKey(50) & 0xff == ord('q'):
                break
            elif i >= 50:
                break
        cap.release()
        cv2.destroyAllWindows()

        row = [roll, name]
        with open(student_details_path + "RegisteredStudents.csv", 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()

        res = "Candidate Added Successfully!!"
        msg.configure(text=res)
        ent.delete(0, 'end')
        ent2.delete(0, 'end')
    else: # if there is any error in the candidate registration.
        if name.isalpha():
            msg.configure(text="Please Enter valid \n ROLL Number")
        else:
            msg.configure(text="Please enter your NAME")


# Front End Code
root = tr.Tk()

root.title("Attendance System")

root.geometry("700x700")
root.configure(background='white')

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

f1 = tr.Frame(root, background="white", width=700, height=170)
f1.grid(row=0, column=0, sticky="nsew")

f2 = tr.Frame(root, background="white", width=700, height=270, highlightbackground="brown", highlightthickness=3)
f2.grid(row=1, column=0, sticky="nsew")

f3 = tr.Frame(root, background="white", width=700, height=260)
f3.grid(row=2, column=0, sticky="nsew")

getRes = tr.Button(root, text="TAKE ATTENDANCE", command=quit, highlightbackground='gold', highlightthickness=5,
                   fg="#000000", width=30, height=2, activebackground="Black", font=('times', 20, 'bold'))
getRes.place(x=190, y=70)

cand = tr.Label(root, text="Add Candidate", fg='brown', width=15, height=1, font=('times', 25, 'italic bold'))
cand.place(x=1, y=154)

txt = tr.Label(f2, text="Name", fg='deeppink', bg="white", width=7, font=('times', 20, 'bold'))
txt.place(x=1, y=70)

ent = tr.Entry(f2, bg="white", width=15, fg="black", font=('times', 20), highlightbackground='gold',
               highlightthickness=1)
ent.place(x=100, y=68)

txt2 = tr.Label(f2, text="Roll no.", fg='deeppink', bg="white", width=7, font=('times', 20, 'bold'))
txt2.place(x=10, y=120)

ent2 = tr.Entry(f2, bg="white", width=15, fg="black", font=('times', 20), highlightbackground='gold',
                highlightthickness=1)
ent2.place(x=100, y=116)

txt3 = tr.Label(f2, text="Notification", width=13, height=3, fg="Deeppink", bg="white", font=('times', 20, ' bold'))
txt3.place(x=380, y=23)

msg = tr.Label(f2, width=25, bg="gray93", height=7, fg="black", activebackground="gray", font=('times', 20))
msg.place(x=400, y=72)

getImg = tr.Button(f2, text="TAKE IMAGES", command=faceDetect, highlightbackground='gold', highlightthickness=5,
                   fg="#000000", width=20, height=2, font=('times', 20, 'bold'))
getImg.place(x=80, y=195)

but1 = tr.Button(f3, text="TRAIN", command=TrainModel, highlightbackground='gold', highlightthickness=5, fg="#000000",
                 width=20, height=2, font=('times', 20, 'bold'))
but1.place(x=80, y=70)

but2 = tr.Button(f3, text="EXIT", command=quit, highlightbackground='red2', fg="#000000", highlightthickness=6,
                 width=15, height=2, font=('times', 20, 'bold'))
but2.place(x=400, y=70)

root.mainloop()
