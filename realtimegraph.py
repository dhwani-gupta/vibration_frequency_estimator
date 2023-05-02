import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def estimate_vibration_frequency_webcam():
    # Open the default camera
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print('Error opening camera')
        return

    # Get the first frame
    ret, frame = cap.read()
    if not ret:
        print('Error reading camera')
        return

    # Convert the frame to grayscale
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize the displacement and time arrays
    disp = []
    t = []

    # Initialize the optical flow parameters
    lk_params = dict(winSize=(15, 15), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Initialize the figure for plotting
    fig, ax = plt.subplots()

    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            print('Error reading camera')
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow between the two frames
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Extract the displacement of the tracked points
        dx = flow[..., 0]
        dy = flow[..., 1]

        # Calculate the total displacement of the tracked points
        disp.append(np.sqrt(np.mean(dx ** 2 + dy ** 2)))

        # Calculate the time elapsed since the start of the video
        t.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)

        # Set the previous frame to the current frame
        prev_gray = gray

        # Estimate the vibration frequency every 100 frames (approx. 3 seconds at 30 fps)
        if len(disp) % 100 == 0:
            # Convert the displacement and time arrays to numpy arrays
            disp = np.array(disp)
            t = np.array(t)

            # Calculate the FFT of the displacement data
            fft = np.fft.fft(disp)
            freq = np.fft.fftfreq(len(disp), t[1] - t[0])

            # Find the peaks in the FFT data
            peaks, _ = find_peaks(np.abs(fft))

            # Clear the displacement and time arrays
            disp = []
            t = []

            # Plot the response graph
            ax.clear()
            ax.plot(freq, np.abs(fft))
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Amplitude')
            ax.set_title('Vibration Response')
            ax.vlines(freq[peaks], 0, np.abs(fft)[peaks], color='r', linestyle='--', label='Peaks')
            ax.legend()
            plt.pause(0.001)

    # Release the camera
    cap.release()

if __name__ == '__main__':
    estimate_vibration_frequency_webcam()
