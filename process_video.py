# importing necessary libraries
import cv2
from ultralytics import YOLO

# initialize the fire detector model
detector = YOLO("./models/fire_detector.pt")

def process(path):
    
    # read
    cap = cv2.VideoCapture("./data/input/test1.mp4")
    
    # read succesful?
    if not cap.isOpened():
        print("Unable to open the video file.")
        exit()
    
    # reading frames
    ret = True
    
    while ret:
        
        ret, frame = cap.read()
        
        results = detector.predict(frame)
        
        for result in results:
            for bbox in result.boxes:
                
                x1, y1, x2, y2 = bbox.xyxy[0]
                
                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                
                class_id = int(bbox.cls)
                class_name = detector.names[class_id]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                cv2.putText(frame, class_name, (x1, y1 + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
                
                cv2.imshow("Detection", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # release the capture object and free memory
    cap.release()
    cv2.destroyAllWindows()

# call the function
process("./data/input/test1.mp4")
    
# uncomment following line to detect and save without visualizing
# detector.predict("./data/input/test1.mp4", save=True)