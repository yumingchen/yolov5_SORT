#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 10:06:51 2021

@author: cym
"""

"""
An example that uses TensorRT's Python api to make inferences.
"""


import ctypes
import time

import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import utils
from sort import Sort
from collections import deque

INPUT_W = 640
INPUT_H = 640

CONF_THRESH = 0.5
IOU_THRESHOLD = 0.3


class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        print('engine_file_path=', engine_file_path)
        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        
    def infer_video(self, img_from_video):
        # Make self the active context, pushing it on top of the context stack.
        self.cfx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        # Copy input image to host buffer
        np.copyto(host_inputs[0], input_image.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]

        return output

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--usb', type=int, default=4, help='usb video serial number')
    parser.add_argument('--side', action='store_true', help='side or front, 摄像头在车道侧面还是正面')
    parser.add_argument('--show', action='store_true', help='是否显示图像')
    parser.add_argument('--save_vedio', action='store_true', help='是否保存output.mp4')
    args = parser.parse_args()
    print(args)
    # load custom plugins
    PLUGIN_LIBRARY = 'yolov5_lib/libmyplugins.so'
    ctypes.CDLL(PLUGIN_LIBRARY)
    engine_file_path  = 'models/yolov5m.engine'

    # load coco labels
    categories = utils.get_classes_name()
    colors = utils.gen_colors(len(categories))
    target_id = [2, 5, 7]
    counts = dict()
    [counts.update({id:0}) for id in target_id]
    # 右向车道的车辆数量
    counts_right = dict()
    [counts_right.update({id: 0}) for id in target_id]
    # 左响车道的车辆计数
    counts_left = dict()
    [counts_left.update({id: 0}) for id in target_id]

    # a  YoLov5TRT instance
    yolov5_wrapper = YoLov5TRT(engine_file_path)
    # 创建跟踪器
    tracker = Sort()
    # 存储中心点
    pts = [deque(maxlen=30) for _ in range(9999)]
    
    WINDOW_NAME = 'TrtYOLODemo'
    title = 'Camera TensorRT YOLO Demo',
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    width, height = 1920, 1080
    if width and height:
        cv2.resizeWindow(WINDOW_NAME, width, height)
    if args.side:
        # cap = cv2.VideoCapture(4)
        cap = cv2.VideoCapture('input/luping_vedio.mp4')
    else:
        cap = cv2.VideoCapture('input/test.mp4')

    if args.save_vedio:
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("[INFO] video size :{}".format(size))
        fps_cur = int(cap.get(cv2.CAP_PROP_FPS))
        print("[INFO] video fps :{}".format(fps_cur))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter("input/output.mp4", fourcc, fps_cur, size, True)
    fps = 0.0
    tic = time.time()
    while(1):
        ret, frame = cap.read()
        if ret is False:
            print('ERROR: Unable to read image!')
            break

        # Do image preprocess
        start_pre = time.time()
        input_image, origin_h, origin_w = utils.preprocess_image(frame, INPUT_W, INPUT_H)
        print(frame.shape)
        end_pre = time.time()
        output = yolov5_wrapper.infer_video(input_image)
        end_infer = time.time()
        # Do postprocess
        nms_pred = utils.post_process(output, origin_h, origin_w, CONF_THRESH, IOU_THRESHOLD, INPUT_W, INPUT_H)

        ################################过滤非目标类别###################################
        target_pred_idx = np.array([False] * len(nms_pred))
        if len(nms_pred) > 0:
            for i, cls_id in enumerate(target_id):
                target_pred_idx = target_pred_idx | (nms_pred[:,5]==cls_id)
            nms_pred = nms_pred[target_pred_idx]
        ################################过滤非目标类别###################################
        end_post = time.time()
        print('the time for pre-processing is [{}]'.format(end_pre - start_pre))
        print('the time for inference is [{}]'.format(end_infer - end_pre))
        print('the time for post-processing is [{}]'.format(end_post - end_infer))

        if args.side:
            line = [(frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[1])]  # 车道侧面
        else:
            line = [(0, frame.shape[0]//2-100), (frame.shape[1], frame.shape[0]//2-100)]  # 车道正面

        cv2.line(frame, line[0], line[1], (0, 255, 0), thickness=5)

        if len(nms_pred) > 0:
            start_track = time.time()
            tracks = tracker.update(nms_pred)  # [x1, y1, x2, y2, id]
            end_track = time.time()

            # Draw rectangles and labels on the original image
            for i in range(len(tracks)):
                bbox = tracks[i][:4]  # 跟踪框坐标
                indexID = int(tracks[i][4])  # 跟踪编号
                center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
                pts[indexID].append(center)
                cls_id = int(nms_pred[i][5])
                score = nms_pred[i][4]
                # 虚拟线圈计数
                if len(pts[indexID]) >= 2:
                    p0, p1 = pts[indexID][-2], pts[indexID][-1]
                    if utils.intersect(p0, p1, line[0], line[1]):
                        counts[cls_id] += 1
                        if args.side:
                            if p1[0] > p0[0]:
                                counts_right[cls_id] += 1
                            else:
                                counts_left[cls_id] += 1
                        else:
                            if p1[1] > p0[1]:
                                counts_left[cls_id] += 1
                            else:
                                counts_right[cls_id] += 1
                if args.show:
                    if int(nms_pred[i][5]) in target_id:
                        utils.plot_one_box( bbox, frame, color=colors[cls_id],
                            label="{}_id_{}:{:.2f}".format(categories[cls_id], indexID, score),
                        )
            end_statistics = time.time()
            print('the time for tracking is [{}]'.format(end_track - start_track))
            print('the time for statistics is [{}]'.format(end_statistics - end_track))
        start_show = time.time()
        if args.show:
            frame = utils.show_statistics_info(frame, counts, counts_left, counts_right, target_id, categories, colors)

        frame = utils.show_fps(frame, fps)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        if args.save_vedio:
            writer.write(frame)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1)==27 or 0xff == ord('q'):
            break
        end_show = time.time()
        print('the time for showing is [{}]'.format(end_show - start_show))
    if args.save_vedio:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    # destroy the instance
    yolov5_wrapper.destroy()


# python cym_yolov5_trt_video.py --usb 4

