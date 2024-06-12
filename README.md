# Deepoint_Object_Detection

This project is derived from a neural network named DeePoint developed by MIT. Apart form DeePoint this model uses another neural networks to detect the objects in the frame and filters the objects to detect which object is being being pointed.

## Prerequisites
The code is tested with `python3.10` with external libraries including:
- `numpy`
- `opencv-python`
- `opencv-contrib-python`
- `torch`
- `torchvision`
- `pytorch-lightning`

Please refer to [environment/pip_freeze.txt](environment/pip_freeze.txt) for the specific versions we used.

## Usage

### Demo
You can download the pretrained model from [here](https://drive.google.com/file/d/1I887Y_G27sPf6QaFfMDTJoHVcTR-pTR_/view?usp=drive_link).
Download it and save the file as `models/weight.ckpt`.

The above weight.ckpt file is provided by the authors of DeePoint.

You can apply the model on your video and visualize the result by running the script below.
```
python src/demo_object_detection.py movie=./demo/JK_demo_1.mp4 lr=l ckpt=./models/weight.ckpt
```

## Important

- The file `src/demo.py` is the original file that is provided by the DeePoint authors.
- Modification in `src/demo.py` lead to `src/demo_object_detection.py`. 
- `src/draw_arrow.py` file is also modified to fit the needs of the model. 

You can find the Original code for the Deepoint Neural Network [here](https://github.com/kyotovision-public/deepoint)