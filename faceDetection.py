import cv2

face_cascade = cv2.CascadeClassifier('DataSets/Faces/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('DataSets/Faces/haarcascade_eye.xml')

img = cv2.imread('DataSets/Faces/dwayne_and_james.jpg') # image is transformed into a numpy array
faces = face_cascade.detectMultiScale(img, 1.3, 5) #returns the position of the detected faces as rect(x,y,w,h)

print('Faces found: ', len(faces))

print('The image height, width, and channel: ',img.shape)  
print('The coordinates of each face detected: ', faces) 

for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) #cv2.rectangle(img, pt1, pt2, color, thickness), NOTE: pt1 os upper left and pt2 is bottom right
    roi_face = img[y:y + h, x:x + w] #find pixel coordinates within the detected face border the region of interest (ROI), because eyes are located on the face
    eyes = eye_cascade.detectMultiScale(roi_face) #returns the position of the detected eyes
    for (ex, ey, ew, eh) in eyes: #loop over all the coordinates eyes returned and draw rectangles around them using Open CV.
        cv2.rectangle(roi_face, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2) #Draw reactangle around the eyes

cv2.imshow('imgage',img) 
cv2.waitKey(0)
cv2.destroyAllWindows() 
