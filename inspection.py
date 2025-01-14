import numpy as np
import sys
def Entropy_label(train_data):
    labels = [row[-1] for row in train_data]
    total_count = len(labels)-1
    count1 = labels.count("1")
    count0 = total_count - count1

    if count1 == 0 or count0 == 0 or total_count==0:
        return 0

    p1 = count1 / total_count
    p0 = count0 / total_count

    training_error=min(count1,count0)/total_count

    entropy = 0
    if p1 > 0:
        entropy -= p1 * np.log2(p1)
    if p0 > 0:
        entropy -= p0 * np.log2(p0)


    return entropy,training_error

def write_to_main(filepath,metrics_output):
    train_data = np.genfromtxt(filepath, delimiter='\t',dtype=str)
    lst=[]
    lst.extend(train_data.tolist())
    a=Entropy_label(lst)
    Entropy=a[0]
    Training_Error=a[1]

    with open(metrics_output, 'w') as f:
        f.write(f"entropy: {Entropy}\n")
        f.write(f"error: {Training_Error}\n")


if __name__ == '__main__':
    train_in = sys.argv[1]
    inspection_output = sys.argv[2]

    write_to_main(train_in, inspection_output)