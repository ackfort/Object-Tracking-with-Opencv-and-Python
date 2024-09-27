import cv2
import time
from datetime import datetime
from tracker import *
import serial  # เพิ่มไลบรารี pySerial

# สร้างการเชื่อมต่อ USART ไปยัง Arduino ผ่านพอร์ตที่ Odroid ใช้ (ปรับให้เป็นพอร์ตที่ถูกต้อง)
arduino = serial.Serial('/dev/ttyACM0', 9600)  # ตรวจสอบพอร์ต serial บน Odroid

# Create tracker object
tracker = EuclideanDistTracker()

# เปิดกล้อง (ใช้ 0 หรือปรับเป็นพอร์ตของกล้องจริงที่เชื่อมต่อ)
cap = cv2.VideoCapture(0)

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# ตัวแปรนับจำนวน ID
detected_ids = set()  # ใช้ set เพื่อเก็บ ID ที่ไม่ซ้ำกัน
last_count = 0  # ตัวแปรเก็บจำนวน ID ก่อนหน้า
start_time = time.time()  # เวลาที่เริ่มต้นนับ

while True:
    
    detection_start_time = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Extract Region of interest
    roi = frame[340: 720, 500: 800]

    # Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    
    for cnt in contours:
        # Calculate area and remove small elements (ปรับค่าขนาดใหญ่ขึ้น)
        area = cv2.contourArea(cnt)
        if area > 5000:  # กรองเฉพาะวัตถุขนาดใหญ่
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    # Object Tracking
    boxes_ids = tracker.update(detections)

    # เพิ่ม ID ลงใน set เพื่อเก็บ ID ที่ไม่ซ้ำกัน
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        detected_ids.add(id)  # เพิ่ม ID ลงใน set
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # ตรวจสอบการเปลี่ยนแปลงจำนวน ID ที่ไม่ซ้ำกัน
    current_count = len(detected_ids)
    if current_count != last_count:  # แสดงเฉพาะเมื่อจำนวนเปลี่ยนแปลง
        print(f'Total Unique IDs: {current_count}')
        
        # เติมศูนย์ด้านหน้าหากมีหลักเดียวหรือสองหลัก
        formatted_count = f"{current_count:03}"  # ทำให้มีความยาว 3 หลัก
        arduino.write(formatted_count.encode())  # ส่งค่า current_count ไปยัง Arduino
        last_count = current_count  # อัปเดตค่าที่ผ่านมาแล้ว

    # ตรวจสอบว่าเวลาได้ผ่านไป 30 วินาทีหรือยัง
    if time.time() - start_time >= 30:
        # บันทึกจำนวน ID และเวลาในไฟล์
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # เวลาในปัจจุบัน
        with open("Implementation-1/detected_ids.txt", "a") as f:
            f.write(f'Time: {current_time}, Total Unique IDs in the last 30 seconds: {current_count}\n')
        
        detected_ids.clear()  # รีเซตค่า ID
        start_time = time.time()  # อัปเดตเวลาเริ่มต้นใหม่

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    
    detection_end_time = time.perf_counter()
    
    # แสดงเวลาในการดำเนินการแต่ละส่วน
    print(f'Time taken for object detection: {detection_end_time - detection_start_time:.4f} seconds')
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
