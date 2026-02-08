import mediapipe as mp
import matplotlib.pyplot as pyplot
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'C:\\Projects\\Hand-Detection\\hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
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

 
image = mp.Image.create_from_file("VertHand.jpg")
with HandLandmarker.create_from_options(options) as HandLandmarker:
    hands = HandLandmarker.detect(image)
    handedNess = hands.handedness
    landmarks = hands.hand_landmarks
    (landmarkListx, landmarkListy) = convertToList(landmarks)
    for i in range(len(fingertips)):
        fingerUp = isFingerUp(landmarkListx,0,fingertips[i],fingermiddles[i])
        print("Finger number "+format(i)+": "+format(fingerUp))
    pyplot.plot(landmarkListx,landmarkListy,'ro')
    pyplot.show()
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
    
