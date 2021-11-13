import csv
import cv2

# 1. Read tracks from CSV
data = {}
current_frame = {
    "frameId": -1
}
with open('scripts/01010003470005401_fast_mot_yolov4.txt', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        frame_id, track_id, left, top, width, height, confidence, _, _, _ = row
        detection = {
            "trackId": track_id,
            "x": round(float(left)),
            "y": round(float(top)),
            "w": round(float(width)),
            "h": round(float(height)),
            "confidence": confidence
        }

        if current_frame["frameId"] != frame_id:
            current_frame = {
                "frameId": frame_id,
                "persons": []
            }
            data[int(frame_id)] = current_frame
        current_frame["persons"].append(detection)

data2 = {}
current_frame = {
    "frameId": -1
}
with open('scripts/01010003470005401_crowd_det.txt', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        frame_id, track_id, left, top, width, height, confidence, _, _, _, _ = row
        detection = {
            "trackId": track_id,
            "x": round(float(left)),
            "y": round(float(top)),
            "w": round(float(width)),
            "h": round(float(height)),
            "confidence": confidence
        }

        if current_frame["frameId"] != frame_id:
            current_frame = {
                "frameId": frame_id,
                "persons": []
            }
            data2[int(frame_id)] = current_frame
        current_frame["persons"].append(detection)        

# 2. Process video
capture = cv2.VideoCapture('scripts/01010003470005401.mp4')
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
fps = capture.get(cv2.CAP_PROP_FPS)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter('01010003470005401_converted.mp4',  0x7634706d, fps, (width, height), True)

ret, frame = capture.read()
frame_id = 0
while ret:
    # if frame_id in data:
    #     persons = data[frame_id]["persons"]
    #     for person in persons:
    #         x = person['x']
    #         y = person['y']
    #         w = person['w']
    #         h = person['h']
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # if frame_id in data2:
    #     persons = data2[frame_id]["persons"]
    #     for person in persons:
    #         x = person['x']
    #         y = person['y']
    #         w = person['w']
    #         h = person['h']
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, f'Frame ID: {frame_id}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
    writer.write(frame)

    if frame_id % 100 == 0:
        print(frame_id)
    frame_id += 1

    ret, frame = capture.read()

capture.release()
writer.release()