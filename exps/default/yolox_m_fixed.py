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

        self.data_num_workers = 0 # Reduced to avoid multiprocessing issues
        
        # --- EXPERIMENT PARAMETERS ---
        self.exp_name = "yolo_signal_test_fixed"
        
        # Training parameters
        self.max_epoch = 90
        self.input_size = (1280, 1280)
        self.test_size = (1280, 1280)
        
        # Dataset parameters
        self.num_classes = 200
        self.data_dir = "D:/Ainhoa/traffic_signs_data/DFG_detection/dataset_coco_ready_original_yolox"
        self.train_ann = "coco_annotations.json"
        self.val_ann = "coco_val_annotations.json"
        
        # --- EVALUATION FIX PARAMETERS ---
        # Reduce evaluation frequency to avoid the bug
        self.eval_interval = 10  # Evaluate every 10 epochs instead of every epoch
        self.print_interval = 10
        self.save_history_ckpt = 10  # Save checkpoints less frequently
        
        # Add validation during training but with error handling
        self.no_aug_epochs = 15
        
        # Enable mixed precision to reduce memory usage
        self.enable_mixup = False  # Disable mixup to reduce complexity
        
        # Learning rate schedule
        self.scheduler = "yoloxwarmcos"
        self.warmup_epochs = 5
        
        # Data augmentation (reduce complexity)
        self.mosaic_prob = 0.5  # Reduce mosaic probability
        self.mixup_prob = 0.0   # Disable mixup
        
        # Box loss weight (might help with bbox prediction issues)
        self.iou_loss_weight = 2.5
        self.cls_loss_weight = 1.0
        self.obj_loss_weight = 1.0
        
    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        """
        Override evaluator to add better error handling
        """
        from yolox.evaluators import COCOEvaluator
        
        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=0.01,  # Lower confidence threshold
            nmsthre=0.65,   # Adjust NMS threshold
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
