# -*- coding:utf-8 -*-
import numpy as np
import os

def main():
    txt_file = open("D:/[1]DB/[3]detection_DB/FDDB/train_label/train_label.txt", "r")

    for i in range(1449):  # number of training dataset
        line = txt_file.readline()
        line = line.split('\n')[0]
        n_box = int(txt_file.readline())	# number of box
        if n_box != 0:
            name = line.split("/")
            name = line.split("/")[0] + "_" \
                + line.split("/")[1] + "_" \
                + line.split("/")[2] + "_" \
                + line.split("/")[3] + "_" \
                + line.split("/")[4]
            write_txt = open("D:/[1]DB/[3]detection_DB/FDDB/train_label/train_label/{}.txt".format(name), "w")
            for j in range(n_box):
                bbox = txt_file.readline()
                bbox = bbox.split('\n')[0]

                xmin = float(bbox.split(' ')[3]) - (float(bbox.split(' ')[0]) / 2)
                ymin = float(bbox.split(' ')[4]) - (float(bbox.split(' ')[1]) / 2)
                xmax = float(bbox.split(' ')[3]) + (float(bbox.split(' ')[0]) / 2)
                ymax = float(bbox.split(' ')[4]) + (float(bbox.split(' ')[1]) / 2)

                write_txt.write(str(xmin))
                write_txt.write(" ")
                write_txt.write(str(ymin))
                write_txt.write(" ")
                write_txt.write(str(xmax))
                write_txt.write(" ")
                write_txt.write(str(ymax))
                write_txt.write(" ")
                write_txt.write(bbox.split(' ')[6])	# class: detect score
                write_txt.write("\n")
                write_txt.flush()
        elif n_box == 0:
            bbox = txt_file.readline()
        if i % 100 == 0:
            print(i+1)

if __name__ == "__main__":
    main()