import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
from keras.models import model_from_json
from tkinter import filedialog


# Load the model
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("emotion_model.h5")

# Initialize the labels for emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy','Neutral', 'Sad', 'Surprise' ]


# Create the tkinter window
window = tk.Tk()
window.title('Emotion Detection')

# Create a canvas to display the video stream
canvas = tk.Canvas(window, width=640, height=480)
canvas.pack(pady=10)

# Create a button to start/stop/getfile the video stream
start_button = tk.Button(window, text='Start', command=lambda: start_video())
start_button.pack(side=tk.LEFT, padx=10, pady=10)

stop_button = tk.Button(window, text='Stop', command=lambda: stop_video())
stop_button.pack(side=tk.LEFT, padx=10, pady=10)

get_file_button = tk.Button(window, text='Get File', command=lambda: get_file())
get_file_button.pack(side=tk.LEFT, padx=10, pady=10)

# Initialize the video capture object
cap = None

# Function to start the video stream
def start_video():
    global cap
    cap = cv2.VideoCapture(0)
    show_frame()

# Function to stop the video stream
def stop_video():
    global cap
    if cap is not None:
        cap.release()
        cap = None
    canvas.delete('all')

def get_file():
    global cap
    # Define which formats of files are allowed to upload
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov"), ("Image Files", "*.jpg *.jpeg *.png *.gif")])
    print("Selected File:", file_path)
    if file_path:
        cap = cv2.VideoCapture(file_path)
        show_frame()


# Function to continuously show frames on the canvas
def show_frame():
    global cap
    if cap is not None:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))  # Resize the image to fit video inside the canvas
        if ret:
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Process each face
            for (x, y, w, h) in faces:
                # Extract the face
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48))
                face = np.expand_dims(face, axis=0)
                face = np.expand_dims(face, axis=-1)
                face = face / 255.0

                # Make prediction
                prediction = model.predict(face)[0]
                max_index = np.argmax(prediction)
                emotion_label = emotion_labels[max_index]

                # Draw rectangle and label on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Convert the frame from BGR to RGB and create a PhotoImage object
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))

            # Show the PhotoImage on the canvas
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.image = photo

            # Call this function again after 10 milliseconds
            window.after(10, show_frame)
        else:
            stop_video()




# Start the tkinter main loop
window.mainloop()
