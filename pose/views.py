import cv2
import mediapipe as mp
import time
from django.shortcuts import render
from .forms import VideoForm

labels = {}

def add_label(name, label_id):
    labels[name] = label_id

add_label("nose", 0)
add_label("left_eye_inner", 1)
add_label("left_eye", 2)
add_label("left_eye_outer", 3)
add_label("right_eye_inner", 4)
add_label("right_eye", 5)
add_label("right_eye_outer", 6)
add_label("left_ear", 7)
add_label("right_ear", 8)
add_label("mouth_left", 9)
add_label("mouth_right", 10)
add_label("left_shoulder", 11)
add_label("right_shoulder", 12)
add_label("left elbow", 13)
add_label("right elbow", 14)
add_label("left_wrist", 15)
add_label("right_wrist", 16)
add_label("left_pinky", 17)
add_label("right_pinky", 18)
add_label("left_index", 19)
add_label("right_index", 20)
add_label("left_thumb", 21)
add_label("right_thumb", 22)
add_label("left_hip", 23)
add_label("right_hip", 24)
add_label("left_knee", 25)
add_label("right_knee", 26)
add_label("left_ankle", 27)
add_label("right_ankle", 28)
add_label("left_heel", 29)
add_label("right_heel", 30)
add_label("left_foot_index", 31)
add_label("right_foot_index", 32)

def get_id(name):
    return labels.get(name)


class PoseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon > 0.0  # Convert to bool
        self.trackCon = trackCon > 0.0 
        self.mppose = mp.solutions.pose
        self.mpdraw = mp.solutions.drawing_utils
        self.pose = self.mppose.Pose(self.mode, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)

    def findpose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        if results.pose_landmarks and draw:
            self.mpdraw.draw_landmarks(img, results.pose_landmarks,
                                       self.mppose.POSE_CONNECTIONS)
        return img
    
    def getposition(self, img, draw=True):
        lmlist = []
        results = self.pose.process(img)  # Corrected variable name
        if results.pose_landmarks:
            for id, lm in enumerate(results.pose_landmarks.landmark):  # Corrected variable name
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 20, (255, 0, 0, 0), cv2.FILLED)
        return lmlist
    
def process(file, id):
    cap = cv2.VideoCapture('videos/' + str(file))  # Capture video from file

    if not cap.isOpened():
        print("Failed to open the video file.")
        return

    ptime = 0
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        
        if not success:
            print("Failed to read frame from the video file.")
            break

        img = detector.findpose(img)
        lmlist = detector.getposition(img, draw=False)
        
        if len(lmlist) != 0 and id is not None and id < len(lmlist):
            cv2.circle(img, (lmlist[id][1], lmlist[id][2]), 20, (255, 0, 0, 0), cv2.FILLED)

        img = cv2.resize(img, (740, 580))
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (70, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow("images", img)
        cv2.waitKey(75)

    cap.release()
    cv2.destroyAllWindows()


def upload(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            id = get_id(form.cleaned_data['Body_part'].lower())
            name = form.cleaned_data['File']
            #print(name)
            print(id)
            form.save()
            if id is not None:
                print(f"The ID for {form.cleaned_data['Body_part']} is {id}")
            else:
                print(f"No ID found for {form.cleaned_data['Body_part']}")
                return render(request, 'upload.html', {'form': form, 'message': 'INVALID ID!'})
                
                # Process the uploaded video
            process(name, id)
            return render(request, 'upload.html', {'form': form, 'message': 'Video processed successfully!'})
    else:
        form = VideoForm()
    return render(request, 'upload.html', {'form': form})
