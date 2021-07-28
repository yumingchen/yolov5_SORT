### yolov5 tensorrt SORT实现车辆计数
#### 本项目可部署到Jetson nona、Jetson tx2、Jetson NX、Jetson agx等设备，故用tensorrt部署。
1. 检测器：yolov5
2. 追踪器：SORT

### Requirement
1. [yolov5](https://github.com/ultralytics/yolov5)
2. [tensorrtx](https://github.com/wang-xinyu/tensorrtx) ：yolov5->tensorrt
### Install
1. clone projetc
    ```
    git clone https://github.com/yumingchen/CVTeamTools
    cd yolov5_SORT
    ```
2. 本项目需部署到Jetson nona、Jetson tx2、Jetson NX、Jetson agx等设备，要求先把pytorch的yolov5模型转为tensorrt模型。
[tensorrtx](https://github.com/wang-xinyu/tensorrtx) 可轻松将yolov5转为tensorrt模型。
详见：https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5
    ```bash
    yolov5m.pt->yolov5m.wts->yolov5m.engine
    ```
### Usage
1. yolov5m.pt->yolov5m.wts->yolov5m.engine，得到engine文件，并放到models文件夹下。
2. tensorrtx编译安装成功后，会生成libmyplugins.so, 将libmyplugins.so复制到yolov5_lib文件下。
    ```bash
    mkdir yolov5_lib
    cd yolov5_lib
    cp xxx/tensorrtx/yolov5/build/libmyplugins.so ./
    xxx/tensorrtx/yolov5/build/libmyplugins.so 
    ```
   
3. parameter
    ```bash
    --side: True：表示摄像头在道路侧面，False：表示摄像头在道路正面。
    ```
4. run
    ```bash
   python yolov5_trt_vedio.py --side
   ```
