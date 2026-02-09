import mediapipe as mp
import cv2 
import time
import matplotlib.pyplot as pyplot
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import winsound as sound


model_path = 'C:\\Projects\\Hand-Detection\\hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)
#Organized in order of index, middle, ring, pinkey
fingertips = [
    8,
    12,
    16,
    20
]
fingermiddles = [
    6,
    10,
    14,
    18
]
def convertToList(landmarks):
    xArr = []
    yArr = []
    for hand in landmarks:
        for landmark in hand:
            #Flips the X axis to transform graph
            xArr.append(landmark.x)
            #Takes compliment of 1(Normalized amount) to make sure image has wrist at the bottom
            yArr.append(1-landmark.y)
            # xArr.append(landmark.x)
            # yArr.append(landmark.y)
    return (xArr,yArr)

def isFingerUp(xArr, hand, fingertip, fingermiddle):
    #Left hand is 0 and right hand is 1
    fingertip+=(hand*21)
    fingermiddle+=(hand*21)
    return xArr[fingertip]>xArr[fingermiddle]
def getTrueIndexes(fingerList):
    indexes = []
    for i in len(fingerList):
        if(fingerList[i]):
            indexes.append(i)
    return indexes
 
cap = cv2.VideoCapture(0)
# image = mp.Image.create_from_file("D_2_Hands.jpg")
detected_note = "NA"
history = deque(maxlen=3)
last_note = "NA"
file = ""
with HandLandmarker.create_from_options(options) as HandLandmarker:
    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            break

        frame=cv2.flip(frame,1)

        frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=frameRGB)
        timestampms = int(time.time()*1000)
        hands = HandLandmarker.detect_for_video(image,timestampms)
        if(len(hands.hand_landmarks)>1):
            #print("Running")
            handedNess = hands.handedness
            landmarks = hands.hand_landmarks
            (landmarkListx, landmarkListy) = convertToList(landmarks)
            fingerList = [] #Right hand and then left hand
            for i in range(len(fingertips)):
                fingerUp =not  isFingerUp(landmarkListx,1,fingertips[i],fingermiddles[i])
                fingerList.append(fingerUp)
                print("Finger number "+format(i)+": "+format(fingerUp))
            print("Left hand")
            for i in range(len(fingertips)):
                fingerUp =  isFingerUp(landmarkListx,0,fingertips[i],fingermiddles[i])
                fingerList.append(fingerUp)
                print("Finger number "+format(i)+": "+format(fingerUp))
            

            history.append(tuple(fingerList))
            stable = max(set(history),key=history.count)

            print(stable)
            if(stable==(True,True,True,False,True,True,True,False)):
                last_note = detected_note
                detected_note = "D"
                file="20239__mtg__sax-alto-single-notes/358378__mtg__sax-alto-d3.wav"
                
            elif(stable==(False,True,False,False,False,False,False,False)):
                last_note = detected_note
                detected_note = "C"
                file="20239__mtg__sax-alto-single-notes/358388__mtg__sax-alto-c4.wav"
            elif(stable==(True,False,False,False,False,False,False,False)):
                last_note = detected_note
                detected_note = "B"
                file="20239__mtg__sax-alto-single-notes/358387__mtg__sax-alto-b3.wav"
            elif(stable==(True,True,False,False,False,False,False,False)):
                last_note = detected_note
                detected_note = "A"
                file="20239__mtg__sax-alto-single-notes/358385__mtg__sax-alto-a3.wav"
            elif(stable==(True,True,True,False,False,False,False,False)):
                last_note = detected_note
                detected_note = "G"
                file="20239__mtg__sax-alto-single-notes/358383__mtg__sax-alto-g3.wav"
            elif(stable==(True,True,True,False,True,False,False,False)):
                last_note = detected_note
                detected_note = "F"
                file="20239__mtg__sax-alto-single-notes/358381__mtg__sax-alto-f3.wav"
            elif(stable==(True,True,True,False,True,True,False,False)):
                last_note = detected_note
                detected_note = "E"
                file="20239__mtg__sax-alto-single-notes/358380__mtg__sax-alto-e3.wav"
            elif(stable==(True,True,True,False,True,True,True,True)):
                last_note = detected_note
                detected_note = "Low C"
                file="20239__mtg__sax-alto-single-notes/358377__mtg__sax-alto-c3.wav"
            elif(stable==(False,False,False,False,False,False,False,False)):
                last_note=detected_note
                detected_note="C#"
                file="20239__mtg__sax-alto-single-notes/358389__mtg__sax-alto-c4.wav"
            elif(stable==(True,True,True,False,True,True,True,True)):
                last_note=detected_note
                detected_note="D#"
                file="20239__mtg__sax-alto-single-notes/358379__mtg__sax-alto-d3.wav"
            elif(stable==(True,True,True,False,False,True,False,False)):
                last_note=detected_note
                detected_note="F#"
                file="20239__mtg__sax-alto-single-notes/358382__mtg__sax-alto-f3.wav"
            elif(stable==(True,True,True,True,False,False,False,False)):
                last_note=detected_note
                detected_note="G#"
                file="20239__mtg__sax-alto-single-notes/358384__mtg__sax-alto-g3.wav"
            elif(stable==(True,False,False,False,True,False,False,False) or stable==(True,False,False,False,False,True,False,False) or stable==(True,False,False,False,True,True,True,False) or stable==(True,False,False,False,True,True,True,False)):
                last_note=detected_note
                detected_note="A#"
                file="20239__mtg__sax-alto-single-notes/358386__mtg__sax-alto-a3.wav"
            
            if detected_note != last_note:
                sound.PlaySound(sound=file,flags=sound.SND_ASYNC|sound.SND_FILENAME)
            cv2.putText(
            frame,
            detected_note,
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            3
            )
        cv2.imshow("Saxophone_Project",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



    
    # pyplot.plot(landmarkListx,landmarkListy,'ro')
    # pyplot.show()
    #print(landmarks)
    #print(hands)
    #print(handedNess)
    # print(landmarks)
    # pyplot.plot(landmarkListx,landmarkListy,'ro')
    # pyplot.show()
    # pyplot.plot([landmarkListx[8]],[[landmarkListy[8]]],'ro')
    # pyplot.show()
    # # pyplot.plot(landmarkListx,landmarkListy,'ro')
    # # pyplot.show()
    # pyplot.plot([landmarkListx[6]],[[landmarkListy[6]]],'ro')
    # pyplot.show()
    #print(type( hands))
    #print(lands)
    
