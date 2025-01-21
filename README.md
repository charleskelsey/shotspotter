# ShotSpotter
ShotSpotter aims to develop an intelligent system capable of automatically recognizing and updating the score of an NBA or any basketball game by analyzing recorded video footage using advanced computer vision techniques. The system will differentiate between various scoring types such as three-pointers, two-pointers, and free throws by processing game footage. The primary goal is to accurately determine the location of the shot, classify the shot type, and update the game score accordingly.

# ShotSpotter Installation Guide

This guide provides instructions on how to set up and use the `Tracker` class for object tracking and detection in video frames. The code is built using the **YOLO** model from Ultralytics and integrates with **supervision** for advanced tracking functionality.

---

## Prerequisites

Ensure the following software and tools are installed:

- Python 3.8 or later
- `pip` for managing Python packages
- A supported GPU (recommended for YOLO model inference)
- OpenCV (for image processing)

---

## Installation

1. **Clone or Download the Repository**

2. **Install Required Dependencies:**

   ```bash
   pip install ultralytics supervision opencv-python-headless numpy pandas
