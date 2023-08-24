import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone
import numpy as np

model=YOLO('yolov8s.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture('p.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0
persondown={}
tracker=Tracker()
counter1=[]

personup={}
counter2=[]

area1=[(494,289),(505,499),(578,496),(530,292)]
area2=[(548,290),(600,496),(637,493),(574,288)]

while True:    
    ret,frame = cap.read()
    if not ret:
        break
#    frame = stream.read()

#    count += 1
#    if count % 3 != 0:
#        continue
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
   
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        
        c=class_list[d]
        if 'person' in c:

            list.append([x1,y1,x2,y2])
       
        
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        result=cv2.pointPolygonTest(np.array(area2,np.int32),((x4,y4)),False)
        if result>=0:
           cv2.circle(frame,(x4,y4),4,(255,0,255),-1)
           cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
           persondown[id]=((cx,cy))
        if id in persondown:   
           result1=cv2.pointPolygonTest(np.array(area1,np.int32),((x4,y4)),False)
           if result1>=0:
              cv2.circle(frame,(x4,y4),4,(255,0,255),-1)
              cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
              if counter1.count(id)==0:
                  counter1.append(id)
        result2=cv2.pointPolygonTest(np.array(area1,np.int32),((x4,y4)),False)
        if result2>=0:
           cv2.circle(frame,(x4,y4),4,(255,0,255),-1)
           cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
           personup[id]=((cx,cy))
        if id in personup:   
           result3=cv2.pointPolygonTest(np.array(area2,np.int32),((x4,y4)),False)
           if result3>=0:
              cv2.circle(frame,(x4,y4),4,(255,0,255),-1)
              cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
              if counter2.count(id)==0:
                  counter2.append(id)

 
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,255,255),2)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,255,255),2)
    studentsout=len(counter1)
    studentsin=len(counter2)
    cvzone.putTextRect(frame,f'StudentsOut:-{studentsout}',(50,60),2,2)
    cvzone.putTextRect(frame,f'StudentsIn:-{studentsin}',(50,160),2,2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

