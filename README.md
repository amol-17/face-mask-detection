# face-mask-detection

Detects face of humans and tell if the person is wearing a mask or not with accuracy of the model. 

## To use the model:

* > pip install -r requirements.txt
* > run **main.py** using **python3 main.py**

## Additional steps for developers who wanted to train the model

1. > Download the [Dataset](https://www.kaggle.com/omkargurav/face-mask-dataset) if you wanted to train the model and save it in folder named **dataset** *(create a new)*
2. > Make sure that the folders inside *dataset* folder are named as: **with_mask** and **without_mask**
3. > You can use any other dataset too for training just remember the _1st_ and _2nd_ point
4. > run **python3 training.py** file.
