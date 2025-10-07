# TinyVGG: Optimized Image Classification Model

Try live at: `https://tinyvgg.streamlit.app`

## 📘 Overview

TinyVGG is a lightweight image classification model inspired by the **VGG16** architecture, re-engineered for efficiency, portability, and high performance.
It achieves **92% classification accuracy** on the CIFAR-10 dataset while maintaining a tiny model size of just **4MB**, making it ideal for deployment on resource-constrained devices such as mobile and embedded systems.

## 🚀 Features

* High Accuracy: Achieved 92% test accuracy on CIFAR-10, matching larger models with fewer parameters.
* Compact Model: Reduced from ~528MB (original VGG16) to 4MB without major performance loss.
* Optimized for Deployment: Lightweight architecture designed for real-time inference and low-latency predictions.
* Preprocessing Pipeline: Built-in image normalization, resizing, and augmentation for consistent input quality.
* Interactive Demo: Easily test model predictions via a Streamlit web interface.

## 🧪 Dataset

**Dataset**: CIFAR-10

**Classes**: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

**Size**: 60,000 images (32×32 pixels, RGB)
