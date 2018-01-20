#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:29:13 2018

@author: angus
"""

### HOW TO USE:
# get a text file of the URLs to parse (see link below for more details)
# setup the below script with the relevant paths etc, then paste something like the following into command line:
#python image_collection.py --urls manet_urls.txt --output manet
# see the following guide by Adrian Rosebrock for more info: https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/

import argparse
import requests
import os

 # used to try to open and delete paths - is this really needed??
#from imutils import paths
#import cv2



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--urls", required=True, help="~/projects/monet_or_manet/manet_urls.txt")
ap.add_argument("-o", "--output", required=True, help="~/projects/monet_or_manet/images/raw/manet")
args = vars(ap.parse_args())
 
# grab the list of URLs from the input file, then initialize the
# total number of images downloaded thus far
rows = open(args["urls"]).read().strip().split("\n")
total = 0

# do one to test script
#r = requests.get('https://upload.wikimedia.org/wikipedia/commons/c/c8/Edouard_Manet_-_At_the_Caf%C3%A9_-_Google_Art_Project.jpg')
#p = os.path.sep.join([args["output"], "{}.jpg".format(str(total).zfill(8))])
#f = open(p, "wb")
#f.write(r.content)
#f.close()


# loop the URLs
for url in rows:
	try:
		# try to download the image
		r = requests.get(url, timeout=60)
 
		# save the image to disk
		p = os.path.sep.join([args["output"], "{}.jpg".format(
			str(total).zfill(8))])
		f = open(p, "wb")
		f.write(r.content)
		f.close()
 
		# update the counter
		print("[INFO] downloaded: {}".format(p))
		total += 1
 
	# handle if any exceptions are thrown during the download process
	except:
		print("[INFO] error downloading {}...skipping".format(p))



























