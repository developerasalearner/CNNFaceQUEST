# CNNFaceQUEST

A face-recognition based attendance system built with a custom Convolutional Neural Network. The system captures face data from a phone camera over Wi-Fi, trains a CNN classifier on the collected faces, and then performs live recognition — automatically logging attendance to a spreadsheet once a face has been identified consistently across consecutive frames.

Developed as a B.Tech Major Project-II (8th semester, 2024–25) in the Department of Computer Science and Engineering, GIET University, Gunupur.

**Team:** Chinmaya Kumar Palo · Rakesh Kumar Nayak · Paresh Sahu
**Supervisor:** Mr. Ranjit Patnaik

---

## Demo

<img width="785" height="524" alt="image" src="https://github.com/user-attachments/assets/68a74385-db0f-4c1e-89d1-a689bca80967" /><img width="805" height="524" alt="image" src="https://github.com/user-attachments/assets/5f474fde-bdf7-47f4-8601-c8dae00a6dec" />



*Three registered individuals detected and identified simultaneously in a single frame.*

---

## Overview

The project covers the full pipeline end to end, from raw data collection to deployed inference:

```
Phone camera (IP Webcam)
        │
        ▼
collect_data.py ──────► images/          100 face crops per person
        │
        ▼
consolidated_data.py ─► CLEAN_DATA/*.p   pickled arrays + labels
        │
        ▼
main__FACE_RECOGNISITION_.ipynb ───────► Final_model_10.h5
        │
        ▼
recognize_2.py  /  convertcsv.py ──────► live prediction + attendance.xlsx
```

Face **detection** is handled classically by an OpenCV Haar cascade; face **identification** is handled by the trained CNN. Separating the two keeps the network's job narrow — it only ever sees a cropped, aligned face region rather than a full frame.

---

## Features

- **Wireless data capture** — pulls frames from an Android phone running the IP Webcam app, so no dedicated camera hardware is needed
- **Automatic face cropping** — Haar cascade isolates the face region before storage, keeping the dataset clean
- **Custom CNN classifier** — a compact architecture trained from scratch on grayscale 100×100 face crops
- **Transfer-learning baseline** — a MobileNetV2 variant included for comparison against the custom model
- **Stability-gated attendance marking** — a face must be predicted consistently across 10 consecutive frames before attendance is recorded, which suppresses single-frame misclassifications
- **Excel export** — attendance is written to `attendance.xlsx` with per-person timestamps

---

## Repository structure

| File | Purpose |
|---|---|
| `collect_data.py` | Captures 100 cropped face images per person from the IP camera stream and saves them to `images/` as `<name>_<index>.jpg` |
| `consolidated_data.py` | Reads every image in `images/`, derives the class label from the filename prefix, and pickles the arrays into `CLEAN_DATA/` |
| `main__FACE_RECOGNISITION_.ipynb` | Primary training notebook — preprocessing, custom CNN definition, training, and model export |
| `new_face_mobilenetv2.ipynb` | Alternative training notebook using MobileNetV2 transfer learning |
| `recognize.py` | Live recognition from the camera stream (earlier 6-class version) |
| `recognize_2.py` | Live recognition matching the shipped 3-class model |
| `convertcsv.py` | Full attendance application — recognition plus stability gating and Excel export |
| `haarcascade_frontalface_default.xml` | Pre-trained OpenCV Haar cascade for frontal face detection |
| `Final_model_10.h5` | Trained custom CNN weights (3 classes) |
| `attendance.xlsx` | Sample attendance output |

---

## Setup

### Requirements

Python 3.8 or newer.

```bash
pip install -r requirements.txt
```

### Camera setup

The scripts read frames from an HTTP snapshot endpoint rather than a local webcam:

1. Install **IP Webcam** (or any equivalent) on an Android phone
2. Connect the phone and the computer to the same Wi-Fi network
3. Start the server in the app and note the address it displays
4. Update the `URL` variable at the top of the relevant script:

```python
URL = 'http://192.168.x.x:8080/shot.jpg'
```

To use a built-in laptop webcam instead, replace the `urllib` frame-fetch block with `cv2.VideoCapture(0)`.

### Paths

The scripts currently contain absolute paths from the original development machines. Update these to point at your local copies before running:

```python
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = load_model("Final_model_10.h5")
```

---

## Usage

### 1. Collect face data

```bash
python collect_data.py
```

Sit in front of the camera. The script counts up to 100 detected faces, then prompts for a name and writes the crops into `images/`. Repeat once per person you want the model to recognise. Varying head angle and lighting during capture meaningfully improves how well the model generalises.

### 2. Build the dataset

```bash
python consolidated_data.py
```

Serialises everything in `images/` into `CLEAN_DATA/images.p` and `CLEAN_DATA/labels.p`. Labels are taken from the filename prefix, so `Chinmaya_42.jpg` becomes class `Chinmaya`.

### 3. Train

Open `main__FACE_RECOGNISITION_.ipynb` (Google Colab works well — upload the two pickle files) and run through. Two things to change if your class count differs from three:

```python
labels = to_categorical(labels, 3)        # ← number of people
model.add(Dense(3, activation='softmax')) # ← same number here
```

The notebook exports `Final_model_10.h5` at the end.

### 4. Run recognition

```bash
python recognize_2.py
```

Draws a bounding box and predicted name over each detected face. Press `q` to quit.

Update the label list to match your own classes and their encoded order — `LabelEncoder` sorts alphabetically, so the list must be in alphabetical order to line up with the model's output indices:

```python
labels = ['CHIKU', 'CHINU', 'MAMA']
```

### 5. Run the attendance application

```bash
python convertcsv.py
```

An interactive prompt controls the session:

- `t` — start or stop the recognition loop
- `e` — export the current attendance record to `attendance.xlsx`
- `q` — quit

---

## Model architecture

### Custom CNN (primary)

Input images are converted to grayscale, histogram-equalised, resized to 100×100, and scaled to `[0, 1]`.

| Layer | Output shape | Parameters |
|---|---|---|
| Conv2D (15 filters, 3×5, ReLU) | 98 × 96 × 15 | 240 |
| MaxPooling2D (2×2) | 49 × 48 × 15 | 0 |
| MaxPooling2D (2×2) | 24 × 24 × 15 | 0 |
| Flatten | 8640 | 0 |
| Dense (512, ReLU) | 512 | 4,424,192 |
| Dense (3, Softmax) | 3 | 1,539 |

**Total parameters:** 4,425,971

Optimiser RMSprop at a learning rate of 0.001, categorical cross-entropy loss, 20 epochs, 10% validation split.

Histogram equalisation is doing real work here — it normalises contrast across frames captured under different lighting, which matters a lot when training data comes from a handheld phone rather than a controlled setup.

### MobileNetV2 (comparison)

ImageNet-pretrained MobileNetV2 backbone with the classification head replaced by `GlobalAveragePooling2D → Dense(200, ReLU) → Dense(3, Softmax)`, trained with Adam at 0.001. Inputs are RGB rather than grayscale. 2,514,787 total parameters.

---

## Results

Dataset: 300 face images across 3 individuals (100 per person), 90/10 train/validation split.

| Model | Train accuracy | Validation accuracy | Validation loss |
|---|---|---|---|
| Custom CNN | 1.000 | 1.000 | 9.02e-06 |
| MobileNetV2 | 0.947 | 0.467 | 7.91 |

The lightweight custom CNN clearly outperforms the transfer-learning approach on this dataset. Two reasons: MobileNetV2 was pretrained at 224×224 and its ImageNet weights transfer poorly when fed 100×100 inputs, and with only 300 samples a 2.5M-parameter backbone with all layers trainable overfits almost immediately — visible in the widening gap between its training and validation curves.

### Honest limitations

The custom model's perfect validation accuracy should not be read as a claim of perfect real-world performance. The 100 images per person were captured in a single continuous session, so consecutive frames are near-duplicates of one another. A random validation split therefore places highly similar images on both sides of the boundary, and the reported validation accuracy measures something closer to memorisation than generalisation.

A more honest evaluation would need:

- A held-out session captured on a different day, under different lighting, as the test set
- Subject-disjoint or session-disjoint splitting rather than a random split
- A rejection mechanism for unknown faces — the softmax layer currently forces every detected face into one of the known classes, so a stranger will always be labelled as someone in the training set

The Haar cascade also inherits well-known constraints: it detects frontal faces reliably but degrades on profile views, heavy occlusion, and poor lighting.

---

## Known issues

- **Hardcoded paths and IPs.** Scripts contain absolute paths and local network addresses from the original machines; these must be edited before use.
- **Label list drift.** `recognize.py` (6 labels), `convertcsv.py` (5 labels), and `recognize_2.py` (3 labels) reflect different training runs. Only `recognize_2.py` matches the shipped `Final_model_10.h5`.
- **`.DS_Store` contamination.** macOS metadata files landing in `images/` create a spurious `.DS` class, which the notebooks strip manually. The included `.gitignore` prevents this recurring.
- **Attendance export shape.** `convertcsv.py` builds the DataFrame from a dict of lists, so uneven timestamp counts across people will raise an error. A long-format record (one row per detection) is the more robust structure.
- **Stability gate is frame-based, not time-based.** The project report describes the trigger as ten seconds of sustained recognition, but the implementation counts ten consecutive frames. Since frame rate varies with network latency and inference speed, the effective dwell time is not fixed. A wall-clock timer would match the documented behaviour.

---

## Possible extensions

- Replace the Haar cascade with an MTCNN or RetinaFace detector for better robustness to pose and lighting
- Move to a face-embedding approach (FaceNet, ArcFace) so new people can be enrolled without retraining the classifier
- Add liveness detection to defeat printed-photo and screen-replay spoofing
- Introduce a confidence threshold so low-certainty predictions are labelled "unknown" instead of being forced into a known class
- Apply data augmentation (rotation, brightness shift, horizontal flip) to enlarge the effective training set

---

## Tech stack

TensorFlow / Keras · OpenCV · NumPy · pandas · scikit-learn · Matplotlib

---

## Documentation

The full project report submitted for evaluation is included at [`docs/Final_Report.pdf`](docs/Final_Report.pdf). It covers the system analysis, high- and low-level design (structure chart, data flow diagram, UML class diagram), unit test cases, and annotated screenshots of every stage of the pipeline.

The report is included as submitted, so its requirements chapters describe the intended system specification rather than the state of this codebase — several items listed there (encryption at rest, load balancing, a graphical interface, administrator error reporting) were specified but not implemented. The **Known issues** section above reflects what the code actually does.

---

## References

1. Godswill, O., Osas, O., Anderson, O., Oseikhuemen, I., & Etse, O. (2018). Automated student attendance management system using face recognition. *International Journal of Educational Research and Information Science*, 5(4), 31–37.
2. Ali, N. S., Alhilali, A. H., Rjeib, H. D., Alsharqi, H., & Al-Sadawi, B. (2022). Automated attendance management systems: systematic literature review. *International Journal of Technology Enhanced Learning*, 14(1), 37–65.
3. Raghuwanshi, A., & Swami, P. D. (2017). An automated classroom attendance system using video based face recognition. *2nd IEEE International Conference on Recent Trends in Electronics, Information & Communication Technology (RTEICT)*, 719–724.
4. Kovelan, P., Thisenthira, N., & Kartheeswaran, T. (2019). Automated attendance monitoring system using IoT. *International Conference on Advancements in Computing (ICAC)*, 376–379.
5. Gupta, N., Sharma, P., Deep, V., & Shukla, V. K. (2020). Automated attendance system using OpenCV. *8th International Conference on Reliability, Infocom Technologies and Optimization (ICRITO)*, 1226–1230.

---

## Note on data

The trained model, the pickled dataset, and the screenshots in the report all contain face images of the project team, captured with their consent for this academic work. If you fork this repository, please train on your own data rather than redistributing these files.

---

## License

Released under the MIT License. See `LICENSE` for details.
