## SPAM or HAM Text classifier with DVC

#### Description
It is a Streamlit web application that allows users to input email text and receive a real-time classification (spam or not spam) using a logistic regression model. The app is designed foreducational purposes, demonstrating basic machine learning concepts.

### Technical Specifications
#### Architecture Overview
Frontend: Streamlit web app
Backend: Python (scikit-learn, pandas)
Model: Logistic Regression (scikit-learn)
Data: Public spam email dataset (e.g., UCI SMS Spam Collection)

### Data Flow   
##### flowchart TD 
- A[User Input Email Text] --> B[Preprocessing] 
- B --> C[Vectorization (e.g., CountVectorizer)] 
- C --> D[Logistic Regression Model] 
- D --> E[Prediction Output]

## How to run?

### STEP 01- Create a conda environment after opening the repository
```bash
conda create -n test1 python=3.11 -y
```

```bash
conda activate test1
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

Now run,
```bash
streamlit run app.py
```



## DVC Commands

git init

dvc init

dvc repro

dvc dag

dvc metrics show

## Git commands

If you are starting a project and you want to use git in your project
```
git init
```
Note: This is going to initalize git in your source code.


OR

You can clone exiting github repo
```
git clone <github_url>
```
Note: Clone/ Downlaod github  repo in your system


Add your changes made in file to git stagging are
```
git add file_name
```
Note: You can given file_name to add specific file or use "." to add everything to staging are


Create commits
```
git commit -m "message"
```

```
git push origin main
```
Note: origin--> contains url to your github repo
main--> is your branch name 

To push your changes forcefully.
```
git push origin main -f
```


To pull  changes from github repo
```
git pull origin main
```
Note: origin--> contains url to your github repo
main--> is your branch name
