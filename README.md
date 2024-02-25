# Project Description for "zircons_recognition"

This project contains PYTHON codes for the following paper. Please consider citing the paper if you find it of any help.
1. Please start with main file: "main.py", training models and predicting zircons types of Jack Hills zircons and global detrital zircons.
2. All other (function) files in "all_models.py" for 11 mainstream models including Decision Tree, GuassianNB, MLP, Bernoulli, Logistic Regressiong, SVM, KNN, Voting, Bagging, Adaboost, TSVM, are called in the function model_train() in main.py.
3. Other figures including fig 1-fig5 in the paper shall be easy to reproduce with the provided function file "plot_figures.py".
4. Models include TSVM is selected as our working model for its best performance.

# Running the project 
## In terminal for Ubuntu or command-line interface for windows
Firstly, clone this project.
```
git clone https://github.com/jiangjilian/zircons_recognition_0625.git
```
Then, to install dependent libraries for our project,run
```
cd zircons_recognition
pip install -r requirements.txt
```
And then, run the main.py to obtain prediction results and our figures.
```
cd code
python main.py
```
Finally, run the plot_figures.py to obtain figures in the article
```
python plot_figures.py
```
## In Pycharm
Install all dependent libraries and directly run main.py, Plot figures.py, and plot_JH_zircons_ratio_with_age.py 
  
