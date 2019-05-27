import csv
import os

def generate_csv():
    images = ["cell-images-for-detecting-malaria/cell_images/Uninfected","cell-images-for-detecting-malaria/cell_images/Parasitized"]
    id = -1

    with open('malaria.csv','w',newline="") as csv_file:
        filewriter = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for i in images:
            id = id + 1
            filenames = os.listdir(i)

            for name in filenames:
                path = i + "/" + name
                filewriter.writerow([path, id])


generate_csv()