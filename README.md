# cvisionProject

This repository is for the CS4501 Intro to Computer Vision Final Project of: Alexander Liang, Milin Patel, Puneeth Uttla, Nico Salafranca

This project will take an image as input, find all of the faces, and attempt to match them to a database. If a match is found, the program will output target's facebook profile. 

This project uses Python 2.7, OpenCV V2.4.13.2 installed from homebrew
Using opencv 2.X because eigenfaces don't work in 3.X




Steps to setup environment (on macOS Sierra v10.12.3)

http://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/

This website describes how to install opencv3. Follow the steps to install homebrew and python2.7. 

In step 4, instead of doing 'brew install opencv3 ...' do 'brew install opencv'. This will install opencv 2.4.13.2 instead of v3.2.0. 

run 'echo /usr/local/opt/opencv/lib/python2.7/site-packages >> /usr/local/lib/python2.7/site-packages/opencv.pth' to link the library to your python installation

open python in the terminal with 'python', the run 
'import cv2'
'print cv2.\_\_version\_\_' 

to make sure you have installed the right version


If you are not on macOS, check the links on the above website. I think he has guides for other OS also. 