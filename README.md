<img width="430" height="465" alt="image" src="https://github.com/user-attachments/assets/78ae8696-3428-4410-8a8e-c5bdd46b53db" /># Indian Sign Language Alphabet Recognition
## Problem Statement
> Build a machine learning system that recognises the indian sign language alphabets from images.
> A hand has [21 landmarks](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS7KoQEE6YAUOeYk8nXpsfFyfUcBMuLfp_dL6lBhFmFtA&s=10.jpg) and each of these landmarks has its own x, y and z coordinates (where x is the horizontal distance, y the vertical distance and z is the extended depth) which in turn makes it 21 x 3 = 63 numerical features to track.
> A feature is the input or the question you give to the machine and the label is the answer of that question. The system checks whether the machine has gotten the answer right by comparing the label and based on how it does on the test, the accuracy score is predicted. 
## Tools Used

- Python
- openCV
- MediaPipe
- Pandas
- Scikit-learn
- Matplotlib
- Random Forest
  
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
│   └── train_model.py
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
## Demo
## Results
## Limitations
## What's next
## Conclusion
## Author
## References
