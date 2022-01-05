from aiohttp import web
import socketio
import cv2
import base64
import imutils
from PIL import Image
import io
from io import StringIO
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 

def extractLips(img,points):
    pList = np.array(points)
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask,pts = [pList],color = (255,255,255))
    img = cv2.bitwise_and(img,mask)
    #cv2.imshow("mask",img)
    return mask

def processImgFrame(image):
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            detections = []
            result = results.multi_face_landmarks[0]
            height, width, channels = image.shape
            detections = []
            for i in LIPS:
                x = result.landmark[i].x * width
                y = result.landmark[i].y * height
                detections.append([int(x),int(y)])
                #cv2.circle(image,(int(x),int(y)),1,(255,255,255),cv2.FILLED)
      
            lips = extractLips(image,detections)
            imgColoredLips = np.zeros_like(lips)
            imgColoredLips[:] = 20,0,157
            imgColoredLips = cv2.bitwise_and(lips,imgColoredLips)
            imgColoredLips = cv2.GaussianBlur(imgColoredLips,(7,7),10)
            imgColoredLips = cv2.addWeighted(image,1,imgColoredLips,0.35,0)
            return imgColoredLips

## creates a new Async Socket IO Server
sio = socketio.AsyncServer()
## Creates a new Aiohttp Web Application
app = web.Application()
# Binds our Socket.IO server to our Web App
## instance
sio.attach(app)

## we can define aiohttp endpoints just as we normally
## would with no change
async def index(request):
    with open('index.html') as f:
        return web.Response(text=f.read(), content_type='text/html')

@sio.on('image')
async def processImage(sid, imgData):
    sbuf = StringIO()
    sbuf.write(imgData)

    # decode and convert into image
    b = io.BytesIO(base64.b64decode(imgData))
    pimg = Image.open(b)

    ## converting RGB to BGR, as opencv standards
    frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

    # Process the image frame
    frame = processImgFrame(frame)
    #frame = imutils.resize(frame, width=500)
    frame = cv2.flip(frame, 1)
    imgencode = cv2.imencode('.jpg', frame)[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    await sio.emit('response_back', stringData)

@sio.on('message')
async def print_message(sid, message):
    print("Socket ID: " , sid)
    print(message)
    await sio.emit('response_back', "stringData")

app.router.add_get('/', index)

if __name__ == '__main__':
    web.run_app(app)