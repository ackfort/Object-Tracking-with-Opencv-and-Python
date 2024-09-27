import cv2
import time
from datetime import datetime
from tracker import *
import serial
import threading

# สร้างการเชื่อมต่อ USART ไปยัง Arduino ผ่านพอร์ตที่ Odroid ใช้
arduino = serial.Serial('/dev/ttyACM0', 9600)

# Create tracker object
tracker = EuclideanDistTracker()

# เปิดกล้อง
cap = cv2.VideoCapture(0)

# ตั้งค่าความละเอียดของกล้อง
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# ตัวแปรนับจำนวน ID
detected_ids = set()
last_count = 0
start_time = time.time()
write_lock = threading.Lock()  # Lock สำหรับการเขียนไฟล์

# ฟังก์ชันสำหรับส่งข้อมูลไปยัง Arduino
def send_to_arduino(count):
    formatted_count = f"{count:03}"
    arduino.write(formatted_count.encode())

# ฟังก์ชันสำหรับเขียนข้อมูลลงไฟล์
def write_to_file(data):
    with write_lock:  # ใช้ lock เพื่อป้องกันการเขียนพร้อมกัน
        with open("Implementation-2/detected_ids.txt", "a") as f:
            f.write(data)

while True:

    detection_start_time = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    roi = frame[340: 720, 500: 800]

    # Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])
    
    # Object Tracking
    boxes_ids = tracker.update(detections)

    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        detected_ids.add(id)
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    current_count = len(detected_ids)
    # ส่งข้อมูลไปยัง Arduino ทุกๆ 1 วินาที
    if current_count != last_count:
        if time.time() - start_time >= 1:  # เช็คว่าผ่านไป 1 วินาทีแล้วหรือไม่
            print(f'Total Unique IDs: {current_count}')
            threading.Thread(target=send_to_arduino, args=(current_count,)).start()
            last_count = current_count
            start_time = time.time()  # เริ่มต้นนับเวลาใหม่

    # ตรวจสอบว่าเวลาได้ผ่านไป 30 วินาทีหรือยัง
    if time.time() - start_time >= 30:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = f'Time: {current_time}, Total Unique IDs in the last 30 seconds: {current_count}\n'
        
        # เริ่ม thread ใหม่สำหรับการเขียนข้อมูลลงไฟล์
        threading.Thread(target=write_to_file, args=(data,)).start()
        
        detected_ids.clear()
        start_time = time.time()

    cv2.imshow("Frame", frame)
   
    detection_end_time = time.perf_counter()
    
        # แสดงเวลาในการดำเนินการแต่ละส่วน
    print(f'Time taken for object detection: {detection_end_time - detection_start_time:.4f} seconds')
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()