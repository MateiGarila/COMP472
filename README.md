# COMP472
COMP 472 project Fall 2024

Team members:
- Ralph Elhage 40131981
- Matei Razvan Garila 40131709

## Dependencies

- Language: Python 3.11.3
- Libraries and Frameworks (from root, use `pip install <dependency>` cmd to install the following):
    - torch
    - torchvision
    - numpy
    - sklearn
    - seaborn
    - matplotlib
    - joblib

## Running guide

1. Clone the repository
2. From terminal, cd into Project_AI/ folder
3. Open Project_AI as the root folder of the project to run the code.
4. Run main.py to install all the training data needed for the models to be trained.
5. Next, open the file corresponding to the AI that you would like to run. The options are:
   - gaussian_naive_bayes.py
   - decision_tree_classifier.py
   - mlp.py
   - cnn.py
6. Run the chosen file

The output of the algorithm will be printed on the terminal. Additionally, the saved model will be saved in its respective folder.
Note that some Python files run both our custom implementations of the algorithms and a variant provided by sklearn. Such files save all the models in the appropriate folders.

**IMPORTANT** note that the CNN saved algorithm could not be included in the repo, as it exceeds github's file size limit.
![image](https://github.com/user-attachments/assets/e6792903-7a62-4956-8ff7-19ce41b7ea04)

PS: an extensive running guide can also be found in our report, in pdf format. Due to the size of the guide, we felt it was more appropriate to have it displayed in pdf form, with screenshots.
