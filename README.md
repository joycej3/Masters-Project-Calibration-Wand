# Masters-Project-Calibration-Wand
This repository contains the code I developed as part of my master's dissertation at Trinity College Dublin. The project investigates a novel method for calculating the intrinsic and extrinsic camera parameters of mobile phone devices using a calibration wand. The goal of this research is to improve the accuracy of parameter calculation to provide more reliable estimations of human pose in movement exercises. 

## Getting Started

Follow the steps below to run the project locally:

### Prerequisites

To run this project, you need the following packages installed. If you don't have them, install them with pip:

pip install numpy
pip install opencv-python
pip install scipy

Also, you need to clone the repository to your local machine. Navigate to the directory where you want to clone the repository and run the following command:
git clone <repository-url>


### Running the code

1. Navigate to the directory where you cloned the repository.
2. Open the Python file with your preferred editor.
3. Edit the 'VideoPath_1' variable to point to the path where your video is located. This video will be used as the input for the camera calibration.
4. Run the code from your terminal using:

python <name-of-the-python-file>.py

vbnet
Copy code

Remember to replace `<name-of-the-python-file>` with the actual filename of the main script.

The code will start processing the frames in the video. It tries to find LED-like objects, calculate their angles, and draw connections between them. 

Press 'Q' on your keyboard at any time to stop the process.

At the end, the script will output some statistics about the processed frames and the video's duration.

### Note

Ensure you update the wand lengths (`mm_wand_lenth_*`) to match the actual physical dimensions of your calibration wand. The current values are just placeholders and may not reflect the actual measurements of your calibration wand.

### Troubleshooting

If you encounter an error that says "Error opening video", check if the path to your video file is correct and the file exists in the mentioned path.
