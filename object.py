import cv2
from tracker import* 

# Create tracker object
tracker = EuclideanDistTracker()


cap=cv2.VideoCapture("highway (2).mp4")
#cap=cv2.VideoCapture(0)
#object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=80)

while True:
    ret, frame=cap.read()

    height,width,_=frame.shape
    #print(height,width)

    #extract roi
    roi=frame[200:720,700:1200]

    #mask
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detection=[]
   
    for cnt in contours:
     #area
     area=cv2.contourArea(cnt)

     if area>500:
         #cv2.drawContours(roi,[cnt],-1,(0,255,0),2)
         x, y, w, h = cv2.boundingRect(cnt)
         
         #print(x,y,w,h)

         detection.append([x,y,w,h])
         #print(detection)

         

#object tracking
    boxes_ids=tracker.update(detection)

    for box_id in boxes_ids:
        x,y,w,h,id=box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
        print("vehicals: ",id)
               

     



    #cv2.imshow("frame",frame)
    cv2.imshow("mask",mask)
    cv2.putText(roi, "Vehicles: " + str(id), (20, 50), 0, 2, (100, 200, 0), 3)
    cv2.imshow("roi",roi)

    if cv2.waitKey(2) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()