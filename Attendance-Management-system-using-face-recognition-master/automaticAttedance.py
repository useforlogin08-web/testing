
import tkinter as tk
from tkinter import *
import os, cv2
import csv
import numpy as np
from PIL import ImageTk, Image
import pandas as pd
import datetime
import time
import tkinter.ttk as tkk
import tkinter.font as font

haarcasecade_path = "haarcascade_frontalface_default.xml"
trainimagelabel_path = "TrainingImageLabel\\Trainner.yml"
trainimage_path = "TrainingImage"
studentdetail_path = "StudentDetails\\StudentDetails.csv"
attendance_path = "Attendance"

def subjectChoose(text_to_speech):

    def FillAttendance():
        sub = tx.get()

        if sub == "":
            t = "Please enter the subject name!!!"
            text_to_speech(t)
            return

        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()

            if not os.path.exists(trainimagelabel_path):
                e = "Model not found, please train images first"
                Notifica.configure(text=e)
                text_to_speech(e)
                return

            recognizer.read(trainimagelabel_path)

            faceCascade = cv2.CascadeClassifier(haarcasecade_path)
            df = pd.read_excel("StudentDetails/StudentDetails.xlsx")

            cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            col_names = ["Enrollment", "Name"]
            attendance = pd.DataFrame(columns=col_names)

            start = time.time()

            while True:
                ret, im = cam.read()

                if not ret:
                    break

                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray, 1.2, 5)

                for (x, y, w, h) in faces:
                    Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

                if conf < 70:
                        if Id in df["Enrollment"].values:
                            name = df.loc[df["Enrollment"] == Id]["Name"].values[0]
                        else:
                            name = "Unknown"
                        attendance.loc[len(attendance)] = [Id, name]

                        cv2.rectangle(im, (x,y),(x+w,y+h),(0,255,0),2)
                        cv2.putText(im, f"{Id}-{name}", (x,y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
                else:
                         cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
                         cv2.putText(im,"Unknown",(x,y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

                cv2.imshow("Filling Attendance...", im)

                if cv2.waitKey(1) == 27:
                    break

                if time.time() - start > 20:
                    break

            attendance = attendance.drop_duplicates(["Enrollment"])

            if not os.path.exists(attendance_path):
                os.makedirs(attendance_path)

            subject_path = os.path.join(attendance_path, sub)

            if not os.path.exists(subject_path):
                os.makedirs(subject_path)

            date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            fileName = os.path.join(subject_path, f"{sub}_{date}.csv")

            attendance.to_csv(fileName, index=False)

            m = "Attendance Filled Successfully"
            Notifica.configure(text=m)
            text_to_speech(m)

            cam.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(e)
            text_to_speech("Error occurred while taking attendance")
            cv2.destroyAllWindows()


    subject = Tk()
    subject.title("Subject")
    subject.geometry("580x320")
    subject.resizable(0,0)
    subject.configure(background="black")

    titl = tk.Label(subject,text="Enter the Subject Name",
                    bg="black",fg="green",font=("arial",25))
    titl.place(x=150,y=10)

    Notifica = tk.Label(subject,text="",bg="yellow",fg="black",
                        width=33,height=2,font=("times",15,"bold"))
    Notifica.place(x=120,y=250)

    sub = tk.Label(subject,text="Enter Subject",width=10,height=2,
                   bg="black",fg="yellow",bd=5,relief=RIDGE,
                   font=("times new roman",15))
    sub.place(x=50,y=100)

    tx = tk.Entry(subject,width=15,bd=5,bg="black",fg="yellow",
                  relief=RIDGE,font=("times",30,"bold"))
    tx.place(x=190,y=100)

    fill_a = tk.Button(subject,text="Fill Attendance",
                       command=FillAttendance,
                       bd=7,font=("times new roman",15),
                       bg="black",fg="yellow",height=2,width=12,
                       relief=RIDGE)
    fill_a.place(x=195,y=170)

    subject.mainloop()
