from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_sqlalchemy import SQLAlchemy
import os
import cv2
import numpy as np


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///videos.db'  
db = SQLAlchemy(app)

for filename in os.listdir(app.config['OUTPUT_FOLDER']):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting {filename}: {e}")

with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')
colors = np.random.uniform(0, 255, size=(len(classes), 3))
class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(255), nullable=False)
    processed_filename = db.Column(db.String(255), nullable=False)

# ...

def process_video(input_path, output_path):
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layers_names = net.getUnconnectedOutLayersNames()

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out_video = None
    writer_opened = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if not writer_opened:
            out_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            writer_opened = True

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(layers_names)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    w = int(detection[2] * frame_width)
                    h = int(detection[3] * frame_height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                label = str(classes[class_id])  # Используем class_id для получения имени класса
                color = colors[class_id]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        out_video.write(frame)  # Запись обработанного кадра в видео

    cap.release()
    if out_video:
        out_video.release()
    cv2.destroyAllWindows()

    input_directory = "/home/zhavkk/PROJE/outputs"
    output_directory = "/home/zhavkk/PROJE/outputs"
    new_resolution = "1920x1080"

    for filename in os.listdir(input_directory):
        if filename.lower().endswith(".mp4"):
            input_file = os.path.join(input_directory, filename)
            filename_without_extension = os.path.splitext(filename)[0]
            output_file = os.path.join(output_directory, f"{filename_without_extension}_new_resolution.mp4")
        
            ffmpeg_command = f"ffmpeg -i {input_file} -s {new_resolution} {output_file}"
            os.system(ffmpeg_command)



    
@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'processed_' + uploaded_file.filename)
            uploaded_file.save(video_path)

            process_video(video_path, output_path)

            new_video = Video(original_filename=uploaded_file.filename, processed_filename=os.path.basename(output_path))
            db.session.add(new_video)
            db.session.commit()

            return redirect(url_for('uploaded_video', video_id=new_video.id))

    return render_template('upload.html')

@app.route('/uploaded/<int:video_id>')
def uploaded_video(video_id):
    video = db.session.get(Video, video_id)
    
    # Получение имени файла с исправленным разрешением
    original_filename = video.original_filename
    filename_without_extension = os.path.splitext(original_filename)[0]
    new_resolution_filename = f"processed_{filename_without_extension}_new_resolution.mp4"
    
    # Генерация нового URL для файла с исправленным разрешением
    new_resolution_video_url = url_for('output_videos', filename=new_resolution_filename)
    
    return render_template('processed.html', video_url=new_resolution_video_url)



@app.route('/output_videos/<filename>')
def output_videos(filename):
    return send_from_directory(os.path.join(os.getcwd(), 'outputs'), filename)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
