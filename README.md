# Gemini-API-Developer-Competition

Project Overview

Our project focuses on advancing the fabrication process of semiconductor devices, especially those based on novel materials beyond conventional silicon-based CMOS technology. As transistors are fundamental components in integrated circuits, their performance heavily relies on the quality of the channels, which are often prepared through exfoliation. Traditionally, the assessment of these channels' quality, including width measurement, is conducted manually under optical or Scanning Electron Microscopes (SEM) using specialized software.

To automate and enhance this process, we have developed a web-based application utilizing the Gemini API and advanced image processing techniques. This tool allows users to upload images of their samples, which are then analyzed to provide immediate feedback on the channel widths and the overall suitability of the samples for device fabrication. Additionally, we have created and trained our own dataset for segmentation models based on the UNet architecture to ensure accurate and reliable results.

This innovation is particularly beneficial for research teams working on nano devices and novel semiconductor materials, offering them a convenient, efficient, and automated solution for quality assessment during the semiconductor fabrication process.


# How to Use the Web Application

# Upload Images:

Click on the "Upload Images for Analysis" button.
Select one or more images from your local machine that you would like to analyze. The images should be representative of the channel samples obtained during the exfoliation step in semiconductor fabrication.
Select Scale Type:

Use the dropdown menu labeled "Select Scale Type" to choose the appropriate scale for your images. The options available are "5 um" (5 micrometers) and "10 um" (10 micrometers).
This step is crucial for accurately calculating the physical dimensions of the features in your images.
Run the Analysis:

After uploading the images and selecting the scale type, the analysis will begin automatically.
The backend will process the images using a pre-trained segmentation model based on the UNet architecture and generate segmentations for the features of interest.
View Results:

# The application will display the following outputs:
Annotated Images: The images with annotated bounding boxes around the detected channels.
Segmented Images: The binary segmented images showing the detected channels.
Results: A summary text describing the analysis performed, including which channel is the best based on the width, length, and intensity criteria.
Analysis: Detailed analysis from the Google Gemini API, providing further insights into the suitability of the channels.
Line Descriptions: A list of measurements for each detected channel, including length, width, and mean intensity.
Download or Save Results:

You can right-click on the annotated and segmented images to download or save them for your records.
Review the text outputs to inform your next steps in the semiconductor fabrication process.
