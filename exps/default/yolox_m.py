#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75

        self.data_num_workers = 0 # <<--- CAMBIAR ESTA LÍNEA (de 4 a 0 o 1)
                                # Si es 0, los datos se cargarán en el proceso principal.
                                # Si es 1, un solo worker cargará los datos.
        
        # --- PARÁMETROS DEL EXPERIMENTO (como YOLOv8) ---
        self.exp_name = "yolo_signal_test" # Equivalente a 'project' + 'name'
                                          # YOLOX creará YOLOX_outputs/yolo_signal_test/best_results_algo/
        
        # self.output_dir = "./YOLOX_outputs" # Directorio de salida por defecto (puedes cambiarlo si quieres)

        # Equivalente a 'epochs' en YOLOv8
        self.max_epoch = 90 # Cambiado de 300 a 100

        # Equivalente a 'imgsz' en YOLOv8
        # Ten en cuenta que YOLOX usa (width, height). Si tu imagen es cuadrada, es (size, size).
        self.input_size = (1280, 1280) # Cambiado de (640, 640) a (1280, 1280)

        # Equivalente a 'cos_lr' en YOLOv8
        self.scheduler = "yoloxwarmcos" # Ya configurado por defecto en yolox_m.py

        # Equivalente a 'save_period' en YOLOv8
        self.save_history_ckpt = 1 # Guardar checkpoint cada 10 épocas

        # --- PARÁMETROS DEL DATASET (ya los tenemos configurados) ---
        self.num_classes = 200 # ¡MUY IMPORTANTE: asegúrate que este número sea el correcto!
        self.data_dir = "D:/Ainhoa/traffic_signs_data/DFG_detection/dataset_coco_ready_original_yolox"
        self.train_ann = "coco_annotations.json"
        self.val_ann = "coco_val_annotations.json"

        # --- OTROS PARÁMETROS DE YOLOX ---
        self.test_size = (1280, 1280) # Ajustar el tamaño de prueba al mismo input_size
        self.eval_interval = 1 # Frecuencia de evaluación (cada cuántas épocas se calcula el mAP)
        self.print_interval = 10 # Frecuencia de impresión de logs en consola
        # self.no_aug_epochs = 15 # Épocas finales sin aumentación (puedes ajustar si quieres)
        
        # La tasa de aprendizaje básica se calcula en base al batch size
        # Si cambias el batch size en el comando, YOLOX ajusta esto automáticamente.
        # self.basic_lr_per_img = 0.00015625 
        
        # Puedes añadir más aumentaciones si lo deseas, pero YOLOX ya tiene mosaic, mixup, hsv, flip, etc.

        # --- FIN MODIFICACIONES ---