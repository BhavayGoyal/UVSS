**Under Vehicle Surveillance System (UVSS)**

An automated Under Vehicle Surveillance System (UVSS) built using computer vision to generate a clear, stitched panoramic view of a vehicle’s undercarriage for security inspection.

🔗 Live Demo: https://uvss-system.streamlit.app/

**Project Description**

Vehicle undercarriage inspection is critical at high-security checkpoints such as airports, borders, military bases, and embassies. Traditional mirror-based inspection is slow, unsafe, and error-prone.

This project implements a complete UVSS pipeline that captures multiple underbody images using planar and fisheye cameras, corrects lens distortion, preprocesses the images, and stitches them into a single seamless panoramic mosaic suitable for visual inspection and future automated analysis.

**Features**

>Multi-image undercarriage capture

>Support for planar and fisheye cameras

>Camera calibration and fisheye undistortion using Scaramuzza’s omnidirectional camera model

>Image preprocessing (cropping, histogram equalization)

>High-quality panoramic stitching using OpenPano (C++)

>ML-ready pipeline for future anomaly detection


**Tech Stack & Tools**

>Programming: Python, C++

>Libraries: OpenCV, OpenPano

>Models: Scaramuzza’s Omnidirectional Camera Model

>Framework: Streamlit (for web demo)

>Domain: Computer Vision, Security & Surveillance Systems

**Future Scope**

* ML-based anomaly detection using convolutional autoencoders

* Pixel-wise heatmaps for suspicious object detection

* Real-time under-vehicle inspection support

