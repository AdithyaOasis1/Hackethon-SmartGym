# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import requests
import json
import random
import face_recognition

FaceCamurl = "http://172.20.10.13:8080/shot.jpg"
TrackingCamUrl = "http://172.20.10.9:8080/shot.jpg"
RepCamUrl = "http://172.20.10.11:8080/shot.jpg"


FaceCamurl = "http://172.20.10.11:8080/shot.jpg"
TrackingCamUrl = "http://172.20.10.9:8080/shot.jpg"
RepCamUrl = "http://172.20.10.11:8080/shot.jpg"

constants = {
	"FacialStage": {
		"XRange": [0, 34],
		"YRange": [0,34]
	},
	"RepStage": {
		"XRange": [144,300],
		"YRange": [144,300]
	}
}

userDB = {
    "4081720": {
        "name": "Adithya"
    }
}

objectToUserIdMapping = {
    "obj22": "3322"
}

# Load a sample picture and learn how to recognize it.
gagan_image = face_recognition.load_image_file("Gagan.png")
gagan_face_encoding = face_recognition.face_encodings(gagan_image)[0]

# Load a second sample picture and learn how to recognize it.
chahak_image = face_recognition.load_image_file("Chahak.png")
chahak_face_encoding = face_recognition.face_encodings(chahak_image)[0]

# Load the third sample picture and learn how to recognize it.
adithya_image = face_recognition.load_image_file("Adithya.png")
adithya_face_encoding = face_recognition.face_encodings(adithya_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    gagan_face_encoding,
    chahak_face_encoding,
    adithya_face_encoding
]
known_face_names = [
    "Gagan Aggarwal",
    "Chahak Sharma",
    "Adithya Oasis"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True



def shouldStartFacial(centroid):
	return PresentInside(constants["FacialStage"], centroid)

def startFacialRecognition():
    print("Starting facial stage")
    img_resp = requests.get(FaceCamurl)
    Dict = {'Chahak Sharma': "4078650", 'Gagan Aggarwal': "4081424", 'Adithya Oasis': "4081720"}
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)
    frame = imutils.resize(frame, width=400)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []
    name = "Unknown"
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
    if name == "Unknown":
        print("Unknown")
        # return None
    print("name" + name)
    userId = Dict[name]
    print("Ended facial stage")
    if not userId:
        userId = "4081720"
    return userId

def mapUserIdWithObjectId(userId, objectId):
    print("objectToUserIdMapping")
    print(objectToUserIdMapping)
    objectToUserIdMapping[objectId] = userId

def shouldStartRepCount(centroid):
	return PresentInside(constants["RepStage"], centroid)

def updateCount(setId, userId, reps):
    with open('DB.json', 'r') as json_file:
        data = json.load(json_file)
        el = [x for x in data if x["setId"] == setId]
        print("Trying to update count with - ")
        print(el)
        if(len(el) == 0):
            saveReps(userId, reps, setId)
        else:
            for index, item in enumerate(data):
                if item.setId == setId:
                    break
            else:
                index = -1
            data[index].reps += 1
            with open('DB.json','w') as json_file:
                json.dump(data, json_file)


def countReps(net, ct, setId, userId):
    print("Starting Rep counting stage")
    (H, W) = (None, None)
    time.sleep(2.0)
    repCounter = 0
    xaxisLeft = 160
    xaxisRight = 300
    hasCompletedHalfRep = False
    diffX = 0
    prevX = -1
    threshold = 30
    exitCounter = 0
    exitThreshold = 20
    breaking = False
    while True:
        # read the next frame from the video stream and resize it
        img_resp = requests.get(RepCamUrl)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, -1)
        frame = imutils.resize(frame, width=400)

        # if the frame dimensions are None, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # construct a blob from the frame, pass it through the network,
        # obtain our output predictions, and initialize the list of
        # bounding box rectangles
        blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
            (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        rects = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # filter out weak detections by ensuring the predicted
            # probability is greater than a minimum threshold
            # if detections[0, 0, i, 2] > args["confidence"]:
            if detections[0, 0, i, 2] > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object, then update the bounding box rectangles list
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                rects.append(box.astype("int"))
                # draw a bounding box surrounding the object so we can
                # visualize it
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 255, 0), 2)
        # update our centroid tracker using the computed set of bounding
        # box rectangles
        objects = ct.update(rects)

        # loop over the tracked objects
        #print("items length - ")
        #print(len(objects.items()))
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            if(centroid[0].item() < xaxisLeft):
                hasCompletedHalfRep = True
    
            if(hasCompletedHalfRep and centroid[0].item() > xaxisRight):
                repCounter += 1
                print("Count updated - " + int(repCounter))
                updateCount(setId, userId, repCounter)
                hasCompletedHalfRep = False
            if(prevX != -1):
                diffX = centroid[0].item() - prevX
            prevX = centroid[0].item()
            if(diffX < threshold):
                exitCounter += 1
            else:
                exitCounter = 0
            print("Current stop time - " + str(exitCounter))
            print("Max stop time - " - exitThreshold)
            if(exitCounter > exitThreshold):
                breaking = True
                break
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            print(repCounter)
        if(breaking == True):
            break
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # do a bit of cleanup
    cv2.destroyAllWindows()
    print("Ended Rep counting stage")

def saveReps(userId, reps, setId):
    data = []
    with open('DB.json', 'r') as json_file:
        data = json.load(json_file)
        newData = {"userId": userId, "reps": reps, "setId": setId}
        data.append(newData)
    with open('DB.json','w') as json_file:
        json.dump(data, json_file)

def PresentInside(Ranges, centroid):
	if centroid[0].item() > Ranges["XRange"][0] and centroid[0].item() < Ranges["XRange"][1] and centroid[1] > Ranges["YRange"][0] and centroid[1] < Ranges["YRange"][1]:
		return True
	return False


def runTracker():
    # initialize our centroid tracker and frame dimensions
    ct = CentroidTracker()
    (H, W) = (None, None)
    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    # initialize the video stream and allow the camera sensor to warmup
    print("[INFO] starting video stream...")
    time.sleep(2.0)
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-c", "--confidence", type=float, default=0.5,
    #     help="minimum probability to filter weak detections")
    # args = vars(ap.parse_args())
    # loop over the frames from the video stream
    while True:
        # read the next frame from the video stream and resize it
        img_resp = requests.get(TrackingCamUrl)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, -1)
        frame = imutils.resize(frame, width=400)

        # if the frame dimensions are None, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # construct a blob from the frame, pass it through the network,
        # obtain our output predictions, and initialize the list of
        # bounding box rectangles
        blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
            (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        rects = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # filter out weak detections by ensuring the predicted
            # probability is greater than a minimum threshold
            # if detections[0, 0, i, 2] > args["confidence"]:
            if detections[0, 0, i, 2] > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object, then update the bounding box rectangles list
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                rects.append(box.astype("int"))

                # draw a bounding box surrounding the object so we can
                # visualize it
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 255, 0), 2)

        # update our centroid tracker using the computed set of bounding
        # box rectangles
        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            objId = "{}".format(objectID)
            # print("objectID")
            # print(objId)
            # print(type(objId))
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            userId = "NOTHING"
            if(shouldStartFacial(centroid)):
                userId = startFacialRecognition()
                mapUserIdWithObjectId(userId, objId)
            if(shouldStartRepCount(centroid)):
                # print("objectToUserIdMapping rep")
                # objId = "{}".format(objectID)
                # print("objectID")
                # print(objId)
                # print(type(objId))
                # print(objectToUserIdMapping)
                # userId = objectToUserIdMapping[objectID]
                # if not userId:
                #     userId = "4081720"
                userId = "4081720"
                setId = random.randint(1000,9999)
                reps = countReps(net, ct, setId, userId)
                # saveReps(userId, reps, setId)
            

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    # saveReps("8", 1)
    runTracker()