# LibTorch Example of SemanticSegmentation  

https://blog.kerrycho.com/Image-Segmentation-Libtorch/

<p align="left">
 <a href="https://github.com/kerry-Cho/SemanticSegmentation-Libtorch/actions"><img alt="GitHub Actions status" src="https://github.com/kerry-Cho/SemanticSegmentation-Libtorch/workflows/C/C++%20CI/badge.svg?branch=master"></a>
</p>

# Current Support CMake
 - [x] Windows
 - [x] Linux
 - [ ] MacOS

  ```
  windows
  mkdir build & cd build
  cmake ..

  linux
  bash build.sh
  ```

# Example
```
#Receive data through commandline.
1. Open cmd.exe

2. training or inference
#example training 
D:\vision>SemanticSegmentation-Libtorch.exe train data_dir 

#example inference
D:\vision>SemanticSegmentation-Libtorch.exe inference image_path model_path 
```

# DataSet
```
# Only Coco data sets are supported.

diretory_root
 cocodataset
  -annotations
   -instances_train2017.json
   -instances_tran2017.json
   -...
  -train2017
   - train images
  -val2017
   - val images
```

# Requirements
 * OpenCV 4.1.1
 * Libtorch 1.4
 * CUDA 10.2
 * Visual studio 2019
 * Windows 10 

Install with Scripts

```
SemanticSegmentation-Libtorch
 - Scripts
    -install.bat
    
Run install.bat
It will download and install automatically.
```

<p align="left">
  <a href="https://github.com/kerry-Cho/SemanticSegmentation-Libtorch"><img alt="GitHub Actions status" src="https://github.com/kerry-Cho/SemanticSegmentation-Libtorch/blob/master/Images/Install.png"></a>
</p>

# Convert Python Model to C++
```
support Only-Resenet backbones
convert.py
Run Python Script
```

# test
  <a href="https://github.com/kerry-Cho/SemanticSegmentation-Libtorch"><img alt="GitHub Actions status" src="https://github.com/kerry-Cho/SemanticSegmentation-Libtorch/blob/master/Images/Samples.png"></a>
</p>


