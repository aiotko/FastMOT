import argparse
import csv
import json
import cv2

# 1. Parse agrugemnts
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('input_mp4', metavar="MP4", help="path to input MP4 file")
parser.add_argument('input_csv', metavar="CSV", help="path to input CSV file")
parser.add_argument('output_json', metavar="JSON", help="path to output JSON file")
args = parser.parse_args()

# 2. Read tracks from CSV
data = []
current_frame = {
    "frameId": -1
}
with open(args.input_csv, newline='') as csvfile:
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
            data.append(current_frame)
        current_frame["persons"].append(detection)

# 3. Read video properties from MP4
capture = cv2.VideoCapture(args.input_mp4)
try:
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
finally:
    capture.release()

# 4. Generate JSON
result = {
    "framesCount": int(frame_count),
    "fps": int(fps),
    "w": width,
    "h": height,
    "data": data
}

with open(args.output_json, 'w') as outfile:
    json.dump(result, outfile, indent=4)
print(f"Convertion complete: {width}x{height} - {fps} fps - {frame_count} frames")