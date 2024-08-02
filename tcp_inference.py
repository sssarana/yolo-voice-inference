import torch
import cv2
import os
import time
from gtts import gTTS
import threading
import queue
from pydub import AudioSegment

# Start tcp server
#subprocess.Popen(["sh", "start_server.sh"])

# Check if GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv5s modele
model = torch.hub.load("ultralytics/yolov5", "yolov5s", device=device)

# Video stream from TCP server
video_stream_url = "tcp://192.168.137.25:34808" # modify the address
cap = cv2.VideoCapture(video_stream_url)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

# Queue for audio announcements
audio_queue = queue.Queue()
# Track last announced objects to avoid repetition
last_announced = set()
# Time interval to reset last announced objects (in seconds)
reset_interval = 10

def play_audio():
    while True:
        text = audio_queue.get()
        if text is None:
            break
        speech_object = gTTS(text=text, lang='en', slow=False)
        speech_object.save("prediction.mp3")
        
        # Convert mp3 to wav
        sound = AudioSegment.from_mp3("prediction.mp3")
        sound.export("prediction.wav", format="wav")
        
        os.system("tinyplay prediction.wav")
        audio_queue.task_done()

# Start audio thread
audio_thread = threading.Thread(target=play_audio, daemon=True)
audio_thread.start()

frame_count = 0
start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    frame_count += 1

    # Process every 5th frame to reduce load
    if frame_count % 5 == 0:
        start_inference_time = time.time()
        
        # Reduce frame resolution for faster processing
        frame_resized = cv2.resize(frame, (640, 480))

        # Inference
        results = model(frame_resized)

        # Process results
        df = results.pandas().xyxy[0]

        # Limit the number of objects to announce per frame
        max_announcements_per_frame = 3
        announcements = 0

        current_time = time.time()
        if current_time - start_time > reset_interval:
            last_announced.clear()
            start_time = current_time

        for name in df['name']:
            if name not in last_announced and announcements < max_announcements_per_frame:
                announcement = f"I see a {name}"
                print(announcement)
                audio_queue.put(announcement)
                last_announced.add(name)
                announcements += 1

        print(f"Inference and processing time: {time.time() - start_inference_time} seconds")

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Stop the audio thread
audio_queue.put(None)
audio_thread.join()
