# auto_label

## Installation
1. Clone this repository.

2. Form github tensorflow/models/research/object_detection download object_detection folder.

3. Replace the ``visualization_utils.py file`` in this given repository into ``object_detection/utils/`` folder.

## Dependencies

1. Tensorflow >= 1.7.0

2. OpenCV = 3.x

3. Keras >= 2.1.3

 For, Python >= 3.5
 
 ## Run 
 1. Put all images in input folder.
    For customized prediction you can use your own pretrained weigths and label_map.pbtxt in a folder.
 
 2. ``python gui.py``.
  
 3. select image folder and model folder as per annotation.
 
 4. The results will be displaced in output Folder.
 
 5. Result images in ``output/images`` and result annotations in ``output/annotations``
