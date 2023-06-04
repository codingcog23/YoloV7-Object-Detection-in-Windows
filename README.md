# YoloV7-Object-Detection-in-Windows

<h2 align="center">YOLOv7: Object Detection for Windows</h2>
<h3 align="left">Comparison of YOLOv7 Weight Detection and YOLOv7-Tiny Weight Detection</h3>
<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/111018114/184507638-e95c42d2-70db-4718-93de-5520933950b8.gif">
  <img width="400" src="https://user-images.githubusercontent.com/111018114/184507880-77504482-c4aa-4e52-b748-9070afc5f02c.gif">
</p>
<h3 align="left">Comparison of YOLOv7 Weight Detection and YOLOv7-Tiny Weight Detection</h3>
<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/111018114/184507538-bb3d2341-fbfa-4c24-bbce-ffc7442d4c2f.jpg">
  <img width="400" src="https://user-images.githubusercontent.com/111018114/184507543-d620c4eb-d846-4cfd-847e-c20a7a1c87c2.jpg">
</p>
<h3 align="left">Comparison of YOLOv7 Weight Detection and YOLOv7-Tiny Weight Detection</h3>
<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/111018114/184508008-6ce5c704-d4cc-4a99-96a6-269ead05960f.gif">
  <img width="400" src="https://user-images.githubusercontent.com/111018114/184508045-c71f3276-718c-429b-92e9-a1121805f58a.gif">
</p>
<h2 align="center">YOLOv7 Overview</h2>

The YOLOv7 paper can be found here.
<p style="text-align: justify;">YOLOv7 is a real-time object detector that is currently revolutionizing the computer vision industry with its incredible features. The official YOLOv7 provides unbelievable speed and accuracy compared to its previous versions. The YOLOv7 weights are trained using Microsoft's COCO dataset, and no pre-trained weights are used.</p>
<p align="center">
  <img width="600" src="https://user-images.githubusercontent.com/111018114/184376228-f9210943-267a-4d54-aa2e-adadbd35b67b.png">
</p>
<h2 align="center">The YOLO Architecture in General</h2>
<p style="text-align: justify;">The YOLO architecture is based on Fully Convolutional Neural Networks (FCNN). However, Transformer-based versions have recently been added to the YOLO family as well. We will discuss Transformer-based detectors in a separate post. For now, let's focus on the FCNN-based YOLO object detectors, which consist of three main components:</p>
<p style="text-align: justify;">
  * Backbone
  * Head
  * Neck
</p>
<p style="text-align: justify;">The Backbone mainly extracts essential features from an image and feeds them to the Head through the Neck. The Neck collects feature maps extracted by the Backbone and creates feature pyramids. Finally, the Head consists of output layers that have the final detections. The following table shows the architectures of YOLOv4, YOLOv4, and YOLOv5.</p>
<h2 align="center">Significance of YOLOv7</h2>
<p style="text-align: justify;">The YOLOv7 paper introduces the following major changes. Let's go through them one by one:</p>
<p style="text-align: justify;">
1. Architectural Reforms
   * E-ELAN (Extended Efficient Layer Aggregation Network)
   * Model Scaling for Concatenation-based Models
2. Trainable BoF (Bag of Freebies)
   * Planned re-parameterized convolution
   * Coarse for auxiliary and Fine for lead loss
</p>
<h2 align="center">Extended Efficient Layer Aggregation</h2>
<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/111018114/184383416-d5eec9c7-ac70-4c69-816b-8b4d2cfc2bf2.png">
</p>
<p style="text-align: justify;">The efficiency of the convolutional layers in the YOLO network's backbone is essential for efficient inference speed. WongKinYiu started down the path of maximal layer efficiency with Cross Stage Partial Networks.</p>

In YOLOv7, the authors build on research that has happened on this topic keeping in mind the amount of memory it takes to keep layers in memory along with the distance that it takes a gradient to back-propagate through the layers - the shorter the gradient, the more powerfully their network will be able to learn. The final layer aggregation they choose is E-ELAN, an extend version of the ELAN computational block.It has been designed by analyzing the following factors that impact speed and accuracy.

1. Memory access cost
2. I/O channel ratio
3. Element wise operation
4. Activations
5. Gradient path 
  
The proposed E-ELAN uses expand, shuffle, merge cardinality to achieve the ability to continuously enhance the learning ability of the network without destroying the original gradient path.

</p>

<h2 align="center">Compound Model Scaling Techniques</h2>

<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/111018114/184385696-4fe6a6e7-c3d3-40db-b851-2aedbd54a95b.png">
</p> 


<p style= 'text-align: justify;'> Object detection models are typically released in a series of models, scaling up and down in size, because different applications require different levels of accuracy and inference speeds. While scaling a model size, the following parameters are considered.
  
1. Resolution ( size of the input image)
2.Width (number of channels)
3. Depth (number of layers)
4. Stage (number of feature pyramids)

Typically, object detection models consider the depth of the network, the width of the network, and the resolution that the network is trained on. In YOLOv7 the authors scale the network depth and width in concert while concatenating layers together. Ablation studies show that this technique keep the model architecture optimal while scaling for different sizes.
</p>


<h2 align="center">Re-parameterization Planning</h2>

<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/111018114/184388086-0bbd64a1-8539-4699-9a00-8ff54d799cd3.png">
</p> 

<p style= 'text-align: justify;'>  Re-parameterization techniques involve averaging a set of model weights to create a model that is more robust to general patterns that it is trying to model. In research, there has been a recent focus on module level re-parameterization where piece of the network have their own re-parameterization strategies. The YOLOv7 authors use gradient flow propagation paths to see which modules in the network should use re-parameterization strategies and which should not.

</p>


<h2 align="center">Auxiliary Head Coarse-to-Fine</h2>

<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/111018114/184388615-b94c0630-19a9-4a7a-80ad-0a838fcaedfb.png">
</p> 

The YOLO network head makes the final predictions for the network, but since it is so far downstream in the network, it can be advantageous to add an auxiliary head to the network that lies somewhere in the middle. While you are training, you are supervising this detection head as well as the head that is actually going to make predictions.

The auxiliary head does not train as efficiently as the final head because there is less network between it an the prediction - so the YOLOv7 authors experiment with different levels of supervision for this head, settling on a coarse-to-fine definition where supervision is passed back from the lead head at different granularities.

<h2 align="center">How to run the YOLOv7 in windows using Anaconda Prompt</h2>


<p style= 'text-align: justify;'> 

### 🚀 Installation

1. clone with YOLOv7 repostery 

```
git clone https://github.com/WongKinYiu/yolov7.git

```
2. Create the new enviornment with python 3.9 to that, open the Anaconda promt and do the followng process 

```
conda create -n yolov7 python=3.9

```
3. Activate the conda enviornment 

```
conda activate yolov7

```

4. Install the conda cuda

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

```

5. Change the path to where you store the YOLOv7 model (while cloning the YOLOv7 github)

```
cd path
```

6. Run the detect.py 

```
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg

```

</p>

<h2 align="center">How to run the YOLOv7 in windows using GitBash</h2>


<p style= 'text-align: justify;'> 

### 🚀 Installation

1. clone with YOLOv7 repostery 

```
git clone https://github.com/WongKinYiu/yolov7.git

```
2. Install the cuda

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

```

3. Change the path to where you store the YOLOv7 model (while cloning the YOLOv7 github)

```
cd path
```

4. Run the detect.py 

```
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg

```
5. How to detect perticular classes in the images:

```
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg --class 0,1,1,3
```

6. How to rund more than one images at a time

```
touch detect.sh

vim detect.sh

#!/bin/bash

for file in inference/images/*.jpg
do 
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source $file 

done
```

</p>
