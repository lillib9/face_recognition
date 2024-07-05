# # import tkinter as tk
# # import cv2
# # import numpy as np
# # from PIL import Image
# # import pandas as pd
# # import datetime
# # import time
# # from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime
# # from sqlalchemy.orm import sessionmaker, declarative_base

# # # Database setup
# # Base = declarative_base()

# # class employees(Base):
# #     _tablename_ = 'employees'
# #     employee_id = Column(Integer, primary_key=True)
# #     name = Column(String)
# #     phone_no = Column(String)
# #     department = Column(String)
# #     joindate = Column(DateTime)
# #     image = Column(LargeBinary)
# #     email = Column(String)

# # class Attended(Base):
# #     _tablename_ = 'attended'
# #     id = Column(Integer, primary_key=True)
# #     employee_id = Column(Integer)
# #     employee_name = Column(String)
# #     attendedtime = Column(DateTime)
# #     image = Column(LargeBinary)

# # class Breach(Base):
# #     _tablename_ = 'breaches'
# #     id = Column(Integer, primary_key=True)
# #     breach_name = Column(String)
# #     image = Column(LargeBinary)

# # # Connect to PostgreSQL database
# # DATABASE_URI = 'postgresql+psycopg2://postgres:Gouthu@123@localhost:5432/step2'
# # engine = create_engine(DATABASE_URI)
# # Base.metadata.create_all(engine)

# # Session = sessionmaker(bind=engine)
# # session = Session()

# # # Tkinter setup
# # window = tk.Tk()
# # window.title("Face_Recogniser")
# # window.geometry('1280x720')
# # window.configure(background='grey')
# # window.grid_rowconfigure(0, weight=1)
# # window.grid_columnconfigure(0, weight=1)

# # message = tk.Label(window, text="Face-Recognition-Based-Attendance-Management-System",
# #                    fg="black", bg="grey", font=('times', 30, 'italic bold underline'))
# # message.place(x=200, y=20)

# # def TrackImages():
# #     recognizer = cv2.face.LBPHFaceRecognizer_create()
# #     recognizer.read("TrainingImageLabel/Trainner.yml")
# #     harcascadePath = "haarcascade_frontalface_default.xml"
# #     faceCascade = cv2.CascadeClassifier(harcascadePath)
    
# #     cam = cv2.VideoCapture(0)
# #     font = cv2.FONT_HERSHEY_SIMPLEX
# #     while True:
# #         ret, im = cam.read()
# #         gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# #         faces = faceCascade.detectMultiScale(gray, 1.2, 5)
# #         for (x, y, w, h) in faces:
# #             cv2.rectangle(im, (x, y, x + w, y + h), (225, 0, 0), 2)
# #             Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
# #             if conf < 60:
# #                 employee = session.query(Employee).filter(Employee.employee_id == Id).first()
# #                 if employee:
# #                     # Check if attendance already logged for today
# #                     today = datetime.datetime.now().date()
# #                     existing_attendance = session.query(Attended).filter(
# #                         Attended.employee_id == Id,
# #                         Attended.attendedtime >= datetime.datetime.combine(today, datetime.time.min),
# #                         Attended.attendedtime <= datetime.datetime.combine(today, datetime.time.max)
# #                     ).first()
                    
# #                     if not existing_attendance:
# #                         ts = time.time()
# #                         attendance_image = cv2.imencode('.jpg', gray[y:y + h, x + x + w])[1].tobytes()
# #                         attendance = Attended(employee_id=Id, employee_name=employee.name,
# #                                               attendedtime=datetime.datetime.fromtimestamp(ts), image=attendance_image)
# #                         session.add(attendance)
# #                         session.commit()
# #                         tt = str(Id) + "-" + employee.name
# #                     else:
# #                         tt = str(Id) + "-" + employee.name + " (Already Logged)"
# #             else:
# #                 tt = "Unknown"
# #                 breach_image = cv2.imencode('.jpg', gray[y:y + h, x + x + w])[1].tobytes()
# #                 breach = Breach(breach_name="Unknown", image=breach_image)
# #                 session.add(breach)
# #                 session.commit()
# #             cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
# #         cv2.imshow('im', im)
# #         if cv2.waitKey(1) == ord('q'):
# #             break
# #     cam.release()
# #     cv2.destroyAllWindows()

# # trackImg = tk.Button(window, text="Track Images", command=TrackImages, width=10, height=1,
# #                      activebackground="Red", font=('times', 15, 'bold'))
# # trackImg.place(x=800, y=500)
# # quitWindow = tk.Button(window, text="Quit", command=window.destroy, width=10, height=1,
# #                        activebackground="Red", font=('times', 15, 'bold'))
# # quitWindow.place(x=1100, y=500)

# # window.mainloop()
# import tkinter as tk
# import cv2
# import numpy as np
# from PIL import Image
# import pandas as pd
# import datetime
# import time
# from urllib.parse import quote_plus
# from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime, ForeignKey
# from sqlalchemy.orm import sessionmaker, declarative_base

# # Database setup
# Base = declarative_base()

# class Employee(Base):
#     __tablename__ = 'employees'
#     employee_id = Column(Integer, primary_key=True)
#     name = Column(String(255), nullable=False)
#     phone_no = Column(String(20))
#     department = Column(String(100))
#     joindate = Column(DateTime)
#     image = Column(LargeBinary)
#     email = Column(String(255))

# class Attended(Base):
#     __tablename__ = 'attended'
#     attended_id = Column(Integer, primary_key=True)
#     employee_id = Column(Integer, ForeignKey('employees.employee_id'), nullable=False)
#     employee_name = Column(String(255), nullable=False)
#     attendedtime = Column(DateTime, nullable=False)
#     image = Column(LargeBinary)

# class Breach(Base):
#     __tablename__ = 'breaches'
#     breach_id = Column(Integer, primary_key=True)
#     breach_time = Column(DateTime, nullable=False)
#     image = Column(LargeBinary)


# username = "postgres" # Your PostgreSQL username
# password = "Gouthu"  # Your PostgreSQL password
# hostname = "localhost"  # Your PostgreSQL host
# port = "5432"  # Your PostgreSQL port
# database = "step2"  # Your database name

# # URL encode the password
# from urllib.parse import quote_plus
# encoded_password = quote_plus(password)






# # Connect to PostgreSQL database

# DATABASE_URI ="postgresql+psycopg2://postgres:Gouthu@localhost:5432/step2"
# print(DATABASE_URI)
# engine = create_engine(DATABASE_URI)
# Base.metadata.create_all(engine)

# Session = sessionmaker(bind=engine)
# session = Session()

# # Tkinter setup
# window = tk.Tk()
# window.title("Face_Recogniser")
# window.geometry('1280x720')
# window.configure(background='grey')
# window.grid_rowconfigure(0, weight=1)
# window.grid_columnconfigure(0, weight=1)

# message = tk.Label(window, text="Face-Recognition-Based-Attendance-Management-System",
#                    fg="black", bg="grey", font=('times', 30, 'italic bold underline'))
# message.place(x=200, y=20)

# def TrackImages():
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read("TrainingImageLabel/Trainner.yml")
#     harcascadePath = "haarcascade_frontalface_default.xml"
#     faceCascade = cv2.CascadeClassifier(harcascadePath)
    
#     cam = cv2.VideoCapture(0)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     while True:
#         ret, im = cam.read()
#         gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#         faces = faceCascade.detectMultiScale(gray, 1.2, 5)
#         for (x, y, w, h) in faces:
#             cv2.rectangle(im, (x, y, x + w, y + h), (225, 0, 0), 2)
#             Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
#             if conf < 60:
#                 employee = session.query(Employee).filter(Employee.employee_id == Id).first()
#                 if employee:
#                     # Check if attendance already logged for today
#                     today = datetime.datetime.now().date()
#                     existing_attendance = session.query(Attended).filter(
#                         Attended.employee_id == Id,
#                         Attended.attendedtime >= datetime.datetime.combine(today, datetime.time.min),
#                         Attended.attendedtime <= datetime.datetime.combine(today, datetime.time.max)
#                     ).first()
                    
#                     if not existing_attendance:
#                         ts = time.time()
#                         attendance_image = cv2.imencode('.jpg', gray[y:y + h, x + x + w])[1].tobytes()
#                         attendance = Attended(employee_id=Id, employee_name=employee.name,
#                                               attendedtime=datetime.datetime.fromtimestamp(ts), image=attendance_image)
#                         session.add(attendance)
#                         session.commit()
#                         tt = str(Id) + "-" + employee.name
#                     else:
#                         tt = str(Id) + "-" + employee.name + " (Already Logged)"
#             else:
#                 tt = "Unknown"
#                 breach_image = cv2.imencode('.jpg', gray[y:y + h, x + x + w])[1].tobytes()
#                 breach = Breach(breach_time=datetime.datetime.now(), image=breach_image)
#                 session.add(breach)
#                 session.commit()
#             cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
#         cv2.imshow('im', im)
#         if cv2.waitKey(1) == ord('q'):
#             break
#     cam.release()
#     cv2.destroyAllWindows()

# trackImg = tk.Button(window, text="Track Images", command=TrackImages, width=10, height=1,
#                      activebackground="Red", font=('times', 15, 'bold'))
# trackImg.place(x=800, y=500)
# quitWindow = tk.Button(window, text="Quit", command=window.destroy, width=10, height=1,
#                        activebackground="Red", font=('times', 15, 'bold'))
# quitWindow.place(x=1100, y=500)

# window.mainloop()
# import tkinter as tk
# import cv2
# import numpy as np
# from PIL import Image
# import pandas as pd
# import datetime
# import time
# from urllib.parse import quote_plus
# from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime, ForeignKey
# from sqlalchemy.orm import sessionmaker, declarative_base

# # Database setup
# Base = declarative_base()

# class Employee(Base):
#     _tablename_ = 'employees'
#     employee_id = Column(Integer, primary_key=True)
#     name = Column(String(255), nullable=False)
#     phone_no = Column(String(20))
#     department = Column(String(100))
#     joindate = Column(DateTime)
#     image = Column(LargeBinary)
#     email = Column(String(255))

# class Attended(Base):
#     _tablename_ = 'attended'
#     attended_id = Column(Integer, primary_key=True)
#     employee_id = Column(Integer, ForeignKey('employees.employee_id'), nullable=False)
#     employee_name = Column(String(255), nullable=False)
#     attendedtime = Column(DateTime, nullable=False)
#     image = Column(LargeBinary)

# class Breach(Base):
#     _tablename_ = 'breaches'
#     breach_id = Column(Integer, primary_key=True)
#     breach_time = Column(DateTime, nullable=False)
#     image = Column(LargeBinary)


# username = "postgres" # Your PostgreSQL username
# password = "Gouthu"  # Your PostgreSQL password
# hostname = "localhost"  # Your PostgreSQL host
# port = "5432"  # Your PostgreSQL port
# database = "step2"  # Your database name

# # URL encode the password
# encoded_password = quote_plus(password)

# # Connect to PostgreSQL database
# DATABASE_URI ="postgresql+psycopg2://postgres:Gouthu@localhost:5432/step2"
# print(DATABASE_URI)
# engine = create_engine(DATABASE_URI)
# Base.metadata.create_all(engine)

# Session = sessionmaker(bind=engine)
# session = Session()

# # Tkinter setup
# window = tk.Tk()
# window.title("Face_Recogniser")
# window.geometry('1280x720')
# window.configure(background='grey')
# window.grid_rowconfigure(0, weight=1)
# window.grid_columnconfigure(0, weight=1)

# message = tk.Label(window, text="Face-Recognition-Based-Attendance-Management-System",
#                    fg="black", bg="grey", font=('times', 30, 'italic bold underline'))
# message.place(x=200, y=20)

# def TrackImages():
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read("TrainingImageLabel/Trainner.yml")
#     harcascadePath = "haarcascade_frontalface_default.xml"
#     faceCascade = cv2.CascadeClassifier(harcascadePath)
    
#     cam = cv2.VideoCapture(0)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     while True:
#         ret, im = cam.read()
#         gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#         faces = faceCascade.detectMultiScale(gray, 1.2, 5)
#         for (x, y, w, h) in faces:
#             cv2.rectangle(im, (x, y, x + w, y + h), (225, 0, 0), 2)
#             Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
#             tt = ""  # Initialize tt with a default value
#             if conf < 60:
#                 employee = session.query(Employee).filter(Employee.employee_id == Id).first()
#                 if employee:
#                     # Check if attendance already logged for today
#                     today = datetime.datetime.now().date()
#                     existing_attendance = session.query(Attended).filter(
#                         Attended.employee_id == Id,
#                         Attended.attendedtime >= datetime.datetime.combine(today, datetime.time.min),
#                         Attended.attendedtime <= datetime.datetime.combine(today, datetime.time.max)
#                     ).first()
                    
#                     if not existing_attendance:
#                         ts = time.time()
#                         attendance_image = cv2.imencode('.jpg', gray[y:y + h, x + x + w])[1].tobytes()
#                         attendance = Attended(employee_id=Id, employee_name=employee.name,
#                                               attendedtime=datetime.datetime.fromtimestamp(ts), image=attendance_image)
#                         session.add(attendance)
#                         session.commit()
#                         tt = str(Id) + "-" + employee.name
#                     else:
#                         tt = str(Id) + "-" + employee.name + " (Already Logged)"
#             else:
#                 tt = "Unknown"
#                 breach_image = cv2.imencode('.jpg', gray[y:y + h, x + x + w])[1].tobytes()
#                 breach = Breach(breach_time=datetime.datetime.now(), image=breach_image)
#                 session.add(breach)
#                 session.commit()
#             cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
#         cv2.imshow('im', im)
#         if cv2.waitKey(1) == ord('q'):
#             break
#     cam.release()
#     cv2.destroyAllWindows()

# trackImg = tk.Button(window, text="Track Images", command=TrackImages, width=10, height=1,
#                      activebackground="Red", font=('times', 15, 'bold'))
# trackImg.place(x=800, y=500)
# quitWindow = tk.Button(window, text="Quit", command=window.destroy, width=10, height=1,
#                        activebackground="Red", font=('times', 15, 'bold'))
# quitWindow.place(x=1100, y=500)

# window.mainloop()
import cv2
import numpy as np
import datetime
import time
import os
import base64
from urllib.parse import quote_plus
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
from sklearn.model_selection import train_test_split

# Database setup
Base = declarative_base()

class Employee(Base):
    __tablename__ = 'employees'
    employee_id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    phone_no = Column(String(20))
    department = Column(String(100))
    joindate = Column(DateTime)
    image = Column(String)  # Store image as base64 text
    email = Column(String(255))

class Attended(Base):
    __tablename__ = 'attended'
    attended_id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(Integer, ForeignKey('employees.employee_id'), nullable=False)
    employee_name = Column(String(255), nullable=False)
    attendedtime = Column(DateTime, nullable=False)
    image = Column(String)  # Store image as base64 text

class Breach(Base):
    __tablename__ = 'breaches'
    breach_id = Column(Integer, primary_key=True, autoincrement=True)
    breach_time = Column(DateTime, nullable=False)
    image = Column(String)  # Store image as base64 text

username = "postgres"  # Your PostgreSQL username
password = "Gouthu"  # Your PostgreSQL password
hostname = "localhost"  # Your PostgreSQL host
port = "5432"  # Your PostgreSQL port
database = "eVision"  # Your database name

# URL encode the password
encoded_password = quote_plus(password)

# Connect to PostgreSQL database
DATABASE_URI = f"postgresql+psycopg2://{username}:{encoded_password}@{hostname}:{port}/{database}"
print(DATABASE_URI)
engine = create_engine(DATABASE_URI)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    faceSamples = []
    ids = []
    
    # Fetch employee images from the database
    employees = session.query(Employee).all()
    
    for employee in employees:
        if employee.image is not None:
            try:
                img_array = np.frombuffer(base64.b64decode(employee.image), np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                
                # Detect the face in the image
                faces = detector.detectMultiScale(img)
                
                if len(faces) == 0:
                    print(f"[WARNING] No face detected for employee ID: {employee.employee_id}, Name: {employee.name}")
                
                # If a face is detected, add the face to the list of samples
                for (x, y, w, h) in faces:
                    faceSamples.append(img[y:y + h, x:x + w])
                    ids.append(employee.employee_id)
            except Exception as e:
                print(f"[ERROR] Failed to process image for employee ID: {employee.employee_id}, Name: {employee.name}. Error: {str(e)}")
        else:
            print(f"[WARNING] No image found for employee ID: {employee.employee_id}, Name: {employee.name}")
    
    if len(faceSamples) == 0:
        print("[ERROR] No faces found in any of the images.")
        return None, None
    
    # Train the recognizer on the face samples
    recognizer.train(faceSamples, np.array(ids))
    
    # Ensure the directory exists
    os.makedirs('TrainingImageLabel', exist_ok=True)
    
    # Save the trained model to a file
    recognizer.write('TrainingImageLabel/Trainner.yml')

    print(f"[INFO] {len(np.unique(ids))} faces trained. Exiting Program")
    
    return faceSamples, ids


def EvaluateModel(faceSamples, ids):
    if faceSamples is None or ids is None:
        print("[ERROR] No faces to evaluate.")
        return

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(faceSamples, ids, test_size=0.3, random_state=42)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(X_train, np.array(y_train))

    correct = 0
    total = len(y_test)

    for img, true_id in zip(X_test, y_test):
        predicted_id, conf = recognizer.predict(img)
        if predicted_id == true_id:
            correct += 1

    accuracy = correct / total
    print(f"[INFO] Model accuracy: {accuracy * 100:.2f}%")


def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    last_id = None
    last_time = time.time()
    duration_threshold = 4 # seconds

    while True:
        ret, im = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y, x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            print(f"Recognized ID: {Id} with confidence: {conf}")  # Debug print
            
            tt = ""  # Initialize tt with a default value
            current_time = time.time()
            if conf < 100:  # Adjust threshold if necessary
                employee = session.query(Employee).filter(Employee.employee_id == Id).first()
                if employee:
                    if last_id == Id and (current_time - last_time) >= duration_threshold:
                        # Check if attendance already logged for today
                        today = datetime.datetime.now().date()
                        existing_attendance = session.query(Attended).filter(
                            Attended.employee_id == Id,
                            Attended.attendedtime >= datetime.datetime.combine(today, datetime.time.min),
                            Attended.attendedtime <= datetime.datetime.combine(today, datetime.time.max)
                        ).first()
                        
                        if not existing_attendance:
                            ts = time.time()
                            attendance_image = cv2.imencode('.jpg', gray[y:y + h, x:x + w])[1].tobytes()
                            attendance_image_base64 = base64.b64encode(attendance_image).decode('utf-8')
                            attendance = Attended(employee_id=Id, employee_name=employee.name,
                                                attendedtime=datetime.datetime.fromtimestamp(ts), image=attendance_image_base64)
                            session.add(attendance)
                            session.commit()
                            tt = str(Id) + "-" + employee.name
                        else:
                            tt = str(Id) + "-" + employee.name + " (Already Logged)"
                    elif last_id != Id:
                        last_id = Id
                        last_time = current_time
                        tt = "Verifying..."
                else:
                    tt = "Unknown (No such employee)"
            else:
                # Only log breach if not already in breaches table
                today = datetime.datetime.now().date()
                existing_breach = session.query(Breach).filter(
                    Breach.breach_time >= datetime.datetime.combine(today, datetime.time.min),
                    Breach.breach_time <= datetime.datetime.combine(today, datetime.time.max)
                ).first()
                
                if last_id is None or (current_time - last_time) >= duration_threshold:
                    if not existing_breach:
                        breach_image = cv2.imencode('.jpg', gray[y:y + h, x:x + w])[1].tobytes()
                        breach_image_base64 = base64.b64encode(breach_image).decode('utf-8')
                        breach = Breach(breach_time=datetime.datetime.now(), image=breach_image_base64)
                        session.add(breach)
                        session.commit()
                        tt = "Unknown"
                    else:
                        tt = "Unknown(came previously)"
                    
                    last_id = None
                    last_time = current_time
                else:
                    tt = "Verifying..."
                
            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
        cv2.imshow('image', im)
        if cv2.waitKey(1) == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    faceSamples, ids = TrainImages()
    EvaluateModel(faceSamples, ids)
    TrackImages()