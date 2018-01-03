#!/usr/bin/env python

import os,sys,shutil
from PIL import Image
import matplotlib.pyplot as plt

def isJournal(file):

    img = Image.open(file)
    img = img.crop((
            img.size[0] / 4,
            400,
            img.size[0] - (img.size[0] / 4),
            img.size[1]))

    hist = img.histogram()
    r,g,b = hist[0:255],hist[255:511],hist[511:767]
    sum_r,sum_g,sum_b = sum(r),sum(g),sum(b)
    red,green,blue = 0,0,0

    if sum_r > 0:
        red = sum(i * w for i, w in enumerate(r)) / sum_r
    if sum_g > 0:
        green = sum(i * w for i, w in enumerate(g)) / sum_g
    if sum_b > 0:
        blue = sum(i * w for i, w in enumerate(b)) / sum_b

    if red > 135 and green > 120 and blue < 95:
        return True

def create_dir(path):
    try:
        os.stat(path)
    except:
        os.mkdir(path)

from paths import *
create_dir(eczema_journal_path)
create_dir(psoriasis_journal_path)
create_dir(eczema_path)
create_dir(psoriasis_path)

valid_files = []
subdirs = []

for subdir, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".jpe") or file.endswith(".bmp"): # Skip all other formats
            valid_files.append(subdir+'/'+file)

    if (not '.' in subdir[-4:]) and ('Patient' in subdir):
        subdirs.append(subdir)

print("Found "+str(len(valid_files))+" pictures and "+str(len(subdirs))+" patients")
num_files = len(valid_files)

print("Organizing pictures...")
num_journals = 0
num_pictues = 0
patient = ""
patient_count = 0
for i,file in enumerate(valid_files):
    sys.stdout.write("\r\t%.2f %% processed" % (float(i)/float(num_files)*100))
    sys.stdout.flush()

    names = file.split("/")
    ending = "."+names[-1].split(".")[-1]
    data_dir = ""
    journ_dir = ""
    if "eczema" in names[1]:
        data_dir = eczema_path
        journ_dir = eczema_journal_path
    elif "psoriasis" in names[1]:
        data_dir = psoriasis_path
        journ_dir = psoriasis_journal_path
    else:
        print("FATAL ERROR: could not place picture in data directory")
        exit(1)

    new_patient = names[-2].replace(" ", "_").lower()
    if patient != new_patient:
        patient = new_patient
        patient_count = 0
    else:
        patient_count += 1

    if patient == "":
        print("FATAL ERROR: no patient found")
        exit(1)

    if isJournal(file):
        #print("journal: ",journ_dir+"/"+patient+"_"+str(patient_count)+ending)
        os.rename(file,    journ_dir+"/"+patient+"_"+str(patient_count)+ending)
        num_journals += 1
    else:
        #print("picture: ",data_dir+"/"+patient+"_"+str(patient_count)+ending)
        os.rename(file,    data_dir+"/"+patient+"_"+str(patient_count)+ending)
        num_pictues += 1

print("\nMoved "+str(num_pictues)+" pictures and "+str(num_journals)+" journals")

print("Deleting old directories...")
shutil.rmtree(eczema_path_old)
shutil.rmtree(psoriasis_path_old)

print("Done!")
exit(0)