![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

# About This Repo
While using Darknet to run the [YOLO](http://arxiv.org/abs/1506.02640) algorithm, I find it a bit hard to adapt the architecture to my own dataset and hard to understand the source code(since there is not a single comment!). Therefore, I forked the original repo and added my own comments for it with instructions here on how to adjust parameters to tune it for your own dataset.

# To Run YOLO with Your Own Dataset
1. Prepare the dataset and annotations.  
2. Modify the parameters in the code.
  1. In scr/yolo.c, change the number of classes that your dataset possesses and their class names. Be sure to add correct paths to in the image reading code snippet.  
  In case you might want to show new classes, run the Python script in data/labels/make_labels.py
  2. In src/yolo_kernels.cu, change the number of classes as it is in the yolo.c.  
  3. Adapt the hyperparameters in the YOLO*.cfg file(the file that specifies the model architecture), including the number of classes, sides and boxes each grid cell predicts(__num__). Also, you need to calculate the output number of the previous fully connected layer, just (5 * num + classes) * side * side.

# References
The original master of Darknet could be found [here](https://github.com/pjreddie/darknet).  
Another fork of Darknet that supports detection in videos resides [here](https://github.com/Guanghan/darknet)  
For those of you who feel inclinded to read the paper, click [here](http://arxiv.org/abs/1506.02640).