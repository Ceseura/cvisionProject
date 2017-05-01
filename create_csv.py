import sys
import os.path

# This is a tiny script to help you creating a CSV file from a face
# database with a similar hierarchy:
#
#  philipp@mango:~/facerec/data/at$ tree
#  .
#  |-- README
#  |-- s1
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  |-- s2
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  ...
#  |-- s40
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#

# Original code taken from https://github.com/Rob-Johnson/opencv-eigenfaces

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "try: python create_csv.py <image_base_path> > <csv_name.csv>"
        sys.exit(1)

    BASE_PATH=sys.argv[1]
    SEPARATOR=";"

    label = 0
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for subdirname in dirnames:
            count = 0
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                # Only want 9 pictures for training data
                if len(filename) < 9:
                    pass
                # Only want files that start with 'pic' and end with 'jpg'
                if filename[:3] == 'pic' and filename[-3:] == 'jpg':
                    abs_path = "%s/%s" % (subject_path, filename)
                    print("%s%s%d" % (abs_path, SEPARATOR, label))
            label = label + 1







