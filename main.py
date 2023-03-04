# importing libraries
###
from facenet_pytorch import MTCNN,InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import cv2
import time
import os



# # initializing MTCNN and InceptionResnetV1 
mtcnn0 = MTCNN(image_size=100, margin=24, keep_all=False, min_face_size=100) # Empty MTCNN intialised: Keep all false means only 1 face will be detected
# Depending on your downstream processing and how fakes can be identified, 
# you may want to add more (or less) of a margin around the detected faces. This is controlled using the margin argument.
# min_face_size denotes the minimum resized input image feeding into P-Net. The lower of this value, the more accurate model is, but in sacrifice of speed
resnet = InceptionResnetV1(pretrained='vggface2').eval() #initializing the class and passing the pretrained model 'Vggface2' and will download if not already

dataset = datasets.ImageFolder('photos') # reads the data from the folder and saves data in the data.pt file
index_to_class = {i:c for c,i in dataset.class_to_idx.items()} # returns the names of the folders that correspond to the images

def collate_fn(x):
    return x[0]

def cal_average(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t           
    avg = sum_num / len(num)
    return avg

def face_detection(loader):
    name_list = [] # list of names corresponding to cropped photos
    embedding_list = [] # list of embedding matrix after conversion from cropped faces to embedding matrix using resnet›››››››
    
    for image, index in loader:
        face, face_prob = mtcnn0(image, return_prob=True) #image is passed into the MTCNN's above and returns the face and probability
        if face is not None and face_prob>0.90:  #if the face is available and prob is > than 0.90 
            # Calculate embedding (unsqueeze to add batch dimension)
            emb = resnet(face.unsqueeze(0))  #then you pass the face into resnet(A CNN that can have thousand of layers and skip 1 or more layers for efficiency)
            embedding_list.append(emb.detach()) #gives the embedding (string of numbers that serves as a unique identifier)
            name_list.append(index_to_class[index])   # gives name list which will contain the names of all the folders
        else:
            print("No face detected")
            continue
    return name_list, embedding_list, face_prob

def minimum_distance(embedding_list, emb):
    dist_list = [] # list of matched distances, minimum distance
    for index, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)

    if len(dist_list) > 0:
        min_dist = min(dist_list) # get minumum dist value 
        min_dist_index = dist_list.index(min_dist) # get minumum dist index
        name = name_list[min_dist_index] # get name corrosponding to minimum dist
        average_minimum_distance.append(min_dist) 
        return average_minimum_distance, name, min_dist, name_list
    else:
        return [], 'No face detected', None, []

def box_frame(name, min_dist,frames, box):
    original_frames = frames.copy() # storing copy of frame before drawing on it

    if min_dist<0.70:
        frames = cv2.putText(frames, name+' '+str(min_dist), (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA)
        #displays name and min distance to the images on the webcam faces
    frames = cv2.rectangle(frames, (int(box[0]),int(box[1])) , (int(box[2]),int(box[3])), (255,0,0), 2)
    return frames, original_frames

def add_box_to_frame(frame, name, min_dist):
    boxes, _ = mtcnn0.detect(frame) # puts a box frame on all faces
    return box_frame(name, min_dist, frame, boxes[0])

def face_classification(frame, average_probability):
    image_cropped, prob = mtcnn0(frame, return_prob=True) #image passed into MTCNN and returns face and probabilty
    if image_cropped is not None and prob > 0.60:
        emb = resnet(image_cropped.unsqueeze(0)).detach() 
        average_probability.append(prob)
        
        average_minimum_distance, name, min_dist, name_list = minimum_distance(embedding_list, emb)

        frame, original_frame = add_box_to_frame(frame, name, min_dist)

        return average_probability, average_minimum_distance, frame, original_frame, name_list, name, min_dist
    

loader = DataLoader(dataset, collate_fn=collate_fn) #converts images into PIL image format for easier processing

name_list, embedding_list, face_prob = face_detection(loader) 

# # save data
data = [embedding_list, name_list] #combines the 2 lists into another list and will save save the data to
torch.save(data, 'data.pt') # data.pt file so that the compuation is not done repeatedly above

# # Using webcam recognize face
load_data = torch.load('data.pt') #Loading the data from data.pt
embedding_list = load_data[0] 
name_list = load_data[1] 

cam = cv2.VideoCapture(0) #when multiple webcams connected, you use the first one

started = time.time() # time.time() is time right now
last_logged = time.time() 
frame_count = 0

average_computation_time_list = []
average_probability = []
average_minimum_distance = []

print("starting!!")
index = 0

while index < 100:
    min_dist = 0
    index += 1
    last_logged = time.time() 
    ret, frame = cam.read() # continously reads and captures images    

    if not ret:
        print("fail to grab frame, try again")
        break

    frame = cv2.resize(frame, (0,0), fx=0.3, fy=0.3) # resize original frame
    # frame = cv2.resize(frame, (960,540)) # resizes frame to (540x960)
    
    result = face_classification(frame, average_probability)
    if result is not None:
        average_probability, average_minimum_distance, frame, original_frame, name_list, name, min_dist = result

    
    now = time.time()
    # displaying model performance below
    frame_count += 1
    computation = now - last_logged
    average_computation_time_list.append(computation)
    
    print("Computation time: ", computation)
    print("Probability that a face has been detected: ", face_prob)
    print("The Minimum distance to your face is: ", min_dist)    
    print()

    k = cv2.waitKey(1)
    if k%256==27: # ESC
        print('Esc pressed, closing...')
        break
        
    elif k%256==32: # space to save image
        print('Enter your name :')
        name = input()
        
        # create directory if not exists
        if not os.path.exists('photos/'+name):
            os.mkdir('photos/'+name)
            
        image_name = "photos/{}/{}.jpg".format(name, int(time.time()))
        cv2.imwrite(image_name, original_frame)
        print(" saved: {}".format(image_name))


    cv2.imshow("IMG", frame)

print("The average computation time per 100 cycles is: ", cal_average(average_computation_time_list))
print("The average probability that a face is detected per 100 cycles is: ", cal_average(average_probability))
print("The average minimum distance to your face per 100 cycles is: ", cal_average(average_minimum_distance))
print()

cam.release()
cv2.destroyAllWindows()


