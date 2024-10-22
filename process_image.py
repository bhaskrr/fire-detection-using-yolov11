# importing necessary libraries
import cv2
from ultralytics import YOLO

# initialize the fire detector model
detector = YOLO("./models/fire_detector.pt")

# reader function
def read_image(path):
    image = cv2.imread(path)
    return image


def detect_and_visualize(img):
    results = detector.predict(img)
    for result in results:
        x1, y1, x2, y2 = result.boxes.xyxy[0]
        
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Detection", img)

        if cv2.waitKey(10000) and 0xFF == ord('q'):
            break

# driver function
def main():
    # image path
    img_path = "./data/input/fire_cctv_1.jpg"
    
    # read and process image
    input_image = read_image(img_path)
    
    # detect
    detect_and_visualize(input_image)


# calling the driver function
main()

# detect and save without visualizing
# detector.predict("./data/input/", save=True)