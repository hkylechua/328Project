"""
This is the script used to combine all collected csv data files into
a single csv file.
"""

import numpy as np
import csv
import time
import labels


# print the available class label (see labels.py)
act_labels = labels.workout_intensity_labels
print(act_labels)

# specify the data files and corresponding activity label
csv_files = ["../data/nothing-Accelerometer.csv", "../data/stretching-Accelerometer.csv", "../data/walking1-Accelerometer.csv", "../data/walking2-Accelerometer.csv", "../data/walking3-Accelerometer.csv", "../data/upstairs1-Accelerometer.csv", "../data/stairmaster1-Accelerometer.csv", "../data/jogging1-Accelerometer.csv", "../data/jogging2-Accelerometer.csv", "../data/jogging3-Accelerometer.csv", "../data/upstairs2-Accelerometer.csv", "../data/stairmaster2-Accelerometer.csv", "../data/absmod1-Accelerometer.csv", "../data/absmod2-Accelerometer.csv", "../data/running1-Accelerometer.csv", "../data/running2-Accelerometer.csv", "../data/running3-Accelerometer.csv", "../data/upstairs3-Accelerometer.csv", "../data/stairmaster3-Accelerometer.csv", "../data/absvig-Accelerometer.csv"]
workout_intensity_list = ["low", "low", "low", "low", "low", "low", "low", "moderate", "moderate", "moderate", "moderate", "moderate", "moderate", "moderate", "vigorous", "vigorous", "vigorous", "vigorous", "vigorous", "vigorous"]

# Specify final output file name. 
output_filename = "../data/all_labeled_data.csv"


all_data = []

zip_list = zip(csv_files, workout_intensity_list)

for f_name, act in zip_list:

    if act in act_labels:
        label_id = act_labels.index(act)
    else:
        print("Label: " + act + " NOT in the activity label list! Check label.py")
        exit()
    print("Process file: " + f_name + " and assign label: " + act + " with label id: " + str(label_id))

    with open(f_name, "r") as f:
        reader = csv.reader(f, delimiter = ",")
        headings = next(reader)
        for row in reader:
            row.append(str(label_id))
            all_data.append(row)


with open(output_filename, 'w',  newline='') as f:
    writer = csv.writer(f)
    writer.writerows(all_data)
    print("Data saved to: " + output_filename)

