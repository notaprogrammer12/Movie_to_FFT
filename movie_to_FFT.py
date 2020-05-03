# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:07:44 2019

@author:

This code will read in a movie file, calculate the brightness of each frame
and create FFT, time and two wav files of the brightness
this is for listening to twinkling of stars

Note: input has to be .mov, output is 32 bit wave file

Function works
Function name: movie_to_FFT(movie_file,frame_per_sec)
Input: movie title and frame rate
Output: two 32 bit wav files of average brightness in movie + time vs 
brightness and FFT of brightness, 

This code works, use it!
"""

from scipy.io.wavfile import write
import numpy as np
import cv2
#from scipy import signal
from scipy import fft
import matplotlib.pyplot as plt

#Function to autocrop an image
#I'm not using because it doesn't sound as good
def autocrop(image, threshold):
    """Crops any edges below or equal to threshold

    Crops blank image to 1x1.

    Returns cropped image.

    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image
## End Function
    
#Function to calculate percentage as an array
def arrayPerentage(input_array):
    input_array = np.array(input_array, dtype=np.float64)  #converts input to 64 bit float to prevent errors
    maxvalue=np.amax(input_array)
    minvalue = np.amin(input_array)
    range_of_array = maxvalue-minvalue
    output_array = 100*(input_array-minvalue)/range_of_array
    #output_array = int(output_array)
    return output_array   #output is 64 bit float

#End function, start main code
#def movie_to_FFT(movie_file='Arcturus',frame_per_sec=60): #Arcturus is 34 sec long
movie_file = 'sirius'    #Input1
cap = cv2.VideoCapture(movie_file+'.mov')    #movie becomes object
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#global array of value of video length
frame_per_sec = 60 #Input 2
vid_len_sec = video_length/frame_per_sec
volume = 1.6   #An arbitrary volume number I have to figure out, 1.6 seems to work well
aveValueFrame = [0]*video_length
aveValueFrame_no_edit = [0]*video_length
highOctaveSound = [0]*video_length*10   #compresses length by a factor of 10

while(cap.isOpened()):          #checks to see if file is opened
    ret, frame = cap.read()     #pulls a single frame out

    if ret: #to close the video when done

        frameNumber=cap.get(cv2.CAP_PROP_POS_FRAMES) # retrieves the current frame number
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converters the image to grey scale, maybe 255?
        #crop_frame=autocrop(gray,100)  #Crops frame to just star, 50-100 is good threshold value, it really doesn't matter
        #crop_frame gives smoother sound while gray gives more squarish sound
       
        #gets pixal value of image * 100
        mean, std = cv2.meanStdDev(gray) #Averages picture to 1 number
        aveValueFrame_no_edit[int(frameNumber)-1]=float(mean)

        cv2.imshow('frame',gray) #shows the current frame in gray scale
        if cv2.waitKey(1) & 0xFF == ord('q'): #waitkey says how long in between frames
            break  #If q is pressed, break out of loop
    else:
        break

cap.release()       #closes the video
cv2.destroyAllWindows() #closes all of the windows

#Setting the volume
max_no_edit = np.amax(aveValueFrame_no_edit)        #Finds max value of gray value
#min_no_edit = np.amin(aveValueFrame_no_edit)        #Finds min value of gray value
scaleFactor = float(volume*((2**32)-1)/max_no_edit)      #should set scale at 90% for 32 bit integer
aveValueFrame=scaleFactor*(np.float64(aveValueFrame_no_edit)-np.mean(aveValueFrame_no_edit)) #removes DC and scales


#turn array into wav file note orignal sample rate was 31 frames/sec
aveValueFrame_auto = np.int32(aveValueFrame) #concatinating to 16 bit int
write(movie_file+'Sound.wav', frame_per_sec, aveValueFrame_auto)

#When trying to get 1000 octive higher computer freezes for 1/2 hour then get MemoryError and it doesn't write file
#looping loop over and over
midOctaveSound=np.concatenate((aveValueFrame,aveValueFrame,aveValueFrame,aveValueFrame,aveValueFrame,aveValueFrame,aveValueFrame,aveValueFrame,aveValueFrame,aveValueFrame), axis=0)
highOctaveSound=np.concatenate((midOctaveSound,midOctaveSound,midOctaveSound,midOctaveSound,midOctaveSound,midOctaveSound,midOctaveSound,midOctaveSound,midOctaveSound,midOctaveSound), axis=0)
highOctaveSound = np.int32(highOctaveSound) #concatinating to 16 bit int
write(movie_file+'Sound100OctaveHigher.wav', frame_per_sec*100, highOctaveSound) #Creates a higher octive sound

#########Plots######
N=len(aveValueFrame) #Number of samples used
T= 1.0/vid_len_sec #Sample spacing

#Plot brightness over time
#aveValueFrame is a 1 dimentional array showing brightness over time, 30 frames/se
time_array=(np.arange(N)+1)*(T+1)/N*vid_len_sec
aveValueFrame_percentage = arrayPerentage(aveValueFrame)
plt.figure(1)
plt.xlabel('Time (Sec)')
plt.ylabel('Percent Brightness')
plt.title('Brightness over Time')
plt.plot(time_array, aveValueFrame_percentage)
plt.grid()
plt.show()


#Caclulate FFT of brightness
#https://scipy.github.io/devdocs/tutorial/fft.html#d-discrete-fourier-transforms
ave_of_aveValueFrame = int(np.mean(aveValueFrame_percentage))
brightnessFFT_Y_Mag=(2.0/N*np.abs(fft(aveValueFrame_percentage-ave_of_aveValueFrame)[0:N//2])) #FFT, but only 1/2 of them
brightnessFFT_Y_dB= 10 * np.log10(brightnessFFT_Y_Mag)  #20 log to get into power dB
freq_FFT = np.linspace(0.0, 1.0/(2.0*T), N//2)
plt.figure(2)
plt.xlabel('Frequency Hz')
plt.ylabel('Brightness (dB)')
plt.title('FFT of Brightness')
plt.plot(freq_FFT, brightnessFFT_Y_dB)
plt.grid()
plt.show()


