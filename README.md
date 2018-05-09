# Tensorflow CPU/GPU installation on Windows 10 64bit     
## Easiest method
- Install [Anaconda](https://www.anaconda.com/download)
- Run `conda install tensorflow-gpu` (This will take care of all dependency installations - Nvidia toolkit, cuda, visual c++ and python library

## Long method
- If you are installing Tensorflow GPU version, check if your NVIDIA GPU is [supported](https://developer.nvidia.com/cuda-gpus) for Tensorflow and has Compute Capability >= 3.0     
- As on 24/3/2017 Tensorflow is supported only on 2.7.x and 3.5.x. So make sure you have this version [Python 64bit](https://www.python.org/downloads/) installed    
- Add Python directory to your [environment variable path](http://www.netinstructions.com/content/images/2016/12/adding-cudnn-to-your-path-tensorflow-windows-7.png) after installation      
            
  
### Tensorflow installation steps   
```python -m pip install tensorflow-gpu # for Tensorflow **GPU** installation```        
```python -m pip install tensorflow # for Tensorflow **CPU** installation```     
- Put the tensorflowvisu.mplstyle in C:\Users\Pratik\AppData\Local\Programs\Python\Python35\Lib\site-packages\matplotlib\mpl-data\stylelib    
- Install [visual studio](https://www.visualstudio.com/downloads) community edition      
- Install [NVIDIA Cuda](https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_win10-exe)      
- Download [cuDNN](https://developer.nvidia.com/cudnn) and put those files where CUDA was installed (merge folders)      
    - cudnn64_5.dll (cuda\bin\cudnn64_5.dll) from the zip archive into C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\        
    - cuda\include\cudnn.h to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include\       
    - cuda\lib\x64\cudnn.lib to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64\        
    - Put C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin to [environment variable path](http://www.netinstructions.com/content/images/2016/12/adding-cudnn-to-your-path-tensorflow-windows-7.png)       

### Done!!   

## Check installation with these lines    

```import tensorflow as tf    

# Below code print out active GPUs   

hello = tf.constant('Hello, TensorFlow!')    
sess = tf.Session()    
a = tf.constant(10)    
b = tf.constant(32)    
print(sess.run(hello))   
print(sess.run(a + b))    

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print(get_available_gpus()) 
```
## Force using CPU instead of GPU
```
with tf.device('/cpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
  
# Creates a session with log_device_placement set to True

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print sess.run(c)    
