import csv
"""
change the path in driving_log.csv to abs path 
"""
if __name__ == '__main__':
    samples = []
    with open('data/driving_log.csv', 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            samples.append(line)
    for item in samples:
        item[0] = '/devdisk/CarND-Behavioral-Cloning-P3/data/'+item[0]
        item[1] = '/devdisk/CarND-Behavioral-Cloning-P3/data/' + item[1]
        item[2] = '/devdisk/CarND-Behavioral-Cloning-P3/data/' + item[2]
    with open('data/driving_log.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(samples)
