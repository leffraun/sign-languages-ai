
# Indian Sign Language Alphabet Recognition 

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?logo=google&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Random Forest](https://img.shields.io/badge/ML-Random%20Forest-2ea44f)

> Problem Statement
> 
> Build a machine learning system that recognises the indian sign language alphabets from images.
> A hand has __21 landmarks__ and each of these landmarks has its own x, y and z coordinates (where x is the horizontal distance, y the vertical distance and z is the extended depth) which in turn makes it __21 x 3 = 63 numerical features__ to track.
> A feature is the input or the question you give to the machine and the label is the answer of that question. The system checks whether the machine has gotten the answer right by comparing the label and based on how it does on the test, the accuracy score is predicted.

<img width="200" height="200" alt="image representing the 21 landmarks of a hand" aria-labe src="https://github.com/user-attachments/assets/78ae8696-3428-4410-8a8e-c5bdd46b53db" style="margin:10%;" />

## Methodology
  ### Brief Info on the tools used:
  - Python
    > Primary programming language used to train the dataset.
  - openCv:
    > Used for image and camera input and basic image processing.
  - MediaPipe:
    > To collect the 63 landmarks of the hands.
  - Pandas
    > To convert the data received from MediaPipe into a csv.
  - Scikit-learn
    > A toolset to train the machine model.
    > It creates the random Forest.
    > It provides accuracy predictions.
  - Matplotlib
    > Used to visualise data and model performance through graphs and plots.
  - Random Forest
    > The machine learning model that we are training using the features (input) and labels (answers) to test machine's knowledge on predicting which alphabet is the given picture.
  
  ### Feature Extraction
  <p>OpenCv captures the 63 features of the hand and MediaPipe converts the given data into csv which is stored in landmarks.csv</p> 
  
## Project Structure
```text
sign-language-ai/
│
├── dataset/
│   └── .gitkeep
│
├── landmarks/
│   └── landmarks.csv
│
├── models/
│   |── hand_landmarker.task
|   |__ sign_language_rf.pkl  
│
├── src/
│   ├── test_landmarker.py
|   ├── inspect_data.py  
│   ├── extract_landmarks.py
│   |── train_model.py
|   └──live_demo.py
│
├── .gitignore
├── README.md
```
## Dataset
  > The original data used to train the machine has been utilised from the [kaggle indian sign language dataset](https://www.kaggle.com/datasets/rushilverma07/indian-sign-language-alphabet-dataset?resource=download).
> In this repository, the dataset has not been given so kindly add it if you do wish to try it.

  > It includes 26 alphabet signs along with an extra { class.   

## Installation
## Run It
```bash
python src/train_model.py

python src/live_demo.py
```
> [!NOTE]
>
> Press Q to exit the video capturing.
## Demo
## Results

| Item | Details |
|---|---:|
|Dataset| Indian Sign Language Alphabet Dataset |
| Hand Landmarks | 21 |
|Model | Random Forest |
|Features per image| 63|
|Classes| 27|
|Test Split| 20%|
|Accuracy | 99.64%|


## Limitations
1. Add a 2 hand image capture 
## What's next
## Conclusion
## Author
## References
