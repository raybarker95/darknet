from ctypes import *
import math
import random
import Tkinter as tk
import tkFileDialog as filedialog
from PIL import Image, ImageTk
import cv2
import os
import subprocess
from glob import glob
import time
live_detector = None
net = None
meta = None

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

# Base file (above) from https://github.com/pjreddie/darknet/blob/master/python/darknet.py
# Functions below added for CITS4402 project

def start_live_detection():
    global live_detector
    live_detector = subprocess.Popen(["./darknet", "detector", "demo", "cfg/coco.data", "cfg/yolov3.cfg", "yolov3.weights"],stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    print("Live detection starting...")

def stop_live_detection():
    global live_detector
    live_detector.terminate()
    print("Live detection stopping...")

def detect_pedestrians(filename):
    global net
    global meta

    # Load net and meta data
    if net is None or meta is None:
        print ("Loading net and meta data")
        net = load_net("cfg/yolov3.cfg", "yolov3.weights", 0)
        meta = load_meta("cfg/coco.data")   

    print ("-----\nRunning detection")
    start_millis = int(round(time.time() * 1000))

    image_in_path = filename
    basename = os.path.basename(image_in_path)
    image_out_path = "output/" + os.path.splitext(basename)[0] + ".png"
    image_resize_path = "resize.png"

    # Resize image maintaining aspect ratio if too large
    im_in = cv2.imread(image_in_path)
    height, width = im_in.shape[:2]
    target_height = 800
    target_width = 1200
    if(height > target_height or width > target_width):
        sf = target_height / float(height)
        if target_width / float(width) < sf:
            sf = target_width / float(width)
        im_in = cv2.resize(im_in, None, fx=sf, fy=sf, interpolation=cv2.INTER_AREA)
        cv2.imwrite(image_resize_path,im_in)
        image_in_path = image_resize_path

    # Perform detection using Darknet YOLO
    detections = detect(net, meta, image_in_path)

    # Prepare image for annotation
    im = cv2.imread(image_in_path)

    # Annotate images with bounding box and detection confidence
    for detection in detections:
        name = detection[0]
        if (name=="person"):
            predict = detection[1]        
            x = int(detection[2][0])
            y = int(detection[2][1])
            w = int(detection[2][2])
            h = int(detection[2][3])            
            cv2.rectangle(im,(x-w/2,y-h/2),(x+w/2,y+h/2),(0,255,0),2)
            cv2.putText(im,str(round(predict,2)),(x-w/2,y-h/2-10),0,0.6,(0,255,0),2)

    # Save annotated image
    cv2.imwrite(image_out_path,im)

    # Update image shown in GUI
    img_out = ImageTk.PhotoImage(Image.open(image_out_path))
    img_box.configure(image=img_out)
    img_box.image = img_out

    duration_millis = int(round(time.time() * 1000)) - start_millis
    print ("Detection complete, took " + str(duration_millis) + " ms")

def myfunction(arg1):
    return None

def run_detection_single():
    image_in_path = filedialog.askopenfilename()
    print image_in_path
    if not image_in_path:
        print "Please select an image file"
        return None
    detect_pedestrians(image_in_path)

def run_detection_batch():
    image_out_path = "annotated.png"
    folder_in_path = filedialog.askdirectory()
    if not folder_in_path:
        print "Please select a directory containing images"
        return None
    
    image_files = glob(folder_in_path + '/*.jpg')
    image_files.extend(glob(folder_in_path + '/*.png'))
    
    for image in image_files:
        detect_pedestrians(image)

        # Update GUI
        root.update_idletasks()

        # Wait until enter is pressed before processing next 
        root.wait_variable(next_batch_image)
        
        
def set_batch_wait(arg):
     next_batch_image.set(not next_batch_image.get())

if __name__ == "__main__":
    
    # Create required folder structure
    if not os.path.exists("output"):
        print "Creating output folder"
        os.makedirs("output")
    
    # Setup GUI
    print ("Loading GUI")
    root = tk.Tk()
    root.title('CITS4402 PROJECT')

    controls = tk.Frame(root)
    controls.pack(side = "left")

    next_batch_image = tk.BooleanVar()
    root.bind('<Key>',set_batch_wait)
    
    run_detection = tk.Button(controls ,text="Load Single Image", command=run_detection_single)
    run_detection.pack(side = "top")

    run_detection = tk.Button(controls ,text="Load Directory", command=run_detection_batch)
    run_detection.pack(side = "top")

    start_live = tk.Button(controls,text="Start Live Feed", command=start_live_detection)
    start_live.pack(side = "top")

    stop_live = tk.Button(controls,text="Stop Live Feed", command=stop_live_detection)
    stop_live.pack(side = "top")

    img = ImageTk.PhotoImage(Image.open('annotated.png'))
    img_box = tk.Label(root, image=img)
    img_box.pack(side="right", fill = "both", expand = "yes")
    
    root.mainloop()
