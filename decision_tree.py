import numpy as np
import sys

class Node:
    def __init__(self, max_depth):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None
        self.max_depth = max_depth

    def process(self, train_data):
        dict_data = {}
        #print("Sample Train Data Row:", train_data[0])
        if np.shape(train_data) == (0,):
            return dict
        for i in range(len(train_data[0])):
            dict_data[i] = [row[i] for row in train_data]
        return dict_data

    def Entropy_label(self, train_data):
        labels = [row[-1] for row in train_data]
        total_count = len(labels)-1
        count1 = labels.count("1")
        count0 = total_count - count1

        if count1 == 0 or count0 == 0 or total_count==0:
            return 0

        ones_count_entropy = count1 / total_count
        zeroes_count_entropy = count0 / total_count

        entropy = 0
        if ones_count_entropy > 0:
            entropy -= ones_count_entropy * np.log2(ones_count_entropy)
        if zeroes_count_entropy > 0:
            entropy -= zeroes_count_entropy * np.log2(zeroes_count_entropy)
        
        return entropy
    
    def get_attribute_names(self, train):
        attribute_names = train[0][:-1]
        return attribute_names

    def train(self, depth, dict_data, train_data, checked_attributes, output, attribute_names):
        if np.shape(train_data) == (0,):
            return output
        if len(checked_attributes) == len(train_data[0]) - 1 or depth == self.max_depth:
            labels = [row[-1] for row in train_data]
            maxi_frequency = max(set(labels), key=labels.count)
            #print(maxi_frequency)
            self.vote = maxi_frequency  
            if labels.count("1")==labels.count("0"):
                self.vote="1"
            return output

        entropy = self.Entropy_label(train_data)
        if entropy == 0:
            labels = [row[-1] for row in train_data]
            maxi_frequency = max(set(labels), key=labels.count)
            self.vote = maxi_frequency
            if labels.count("1")==labels.count("0"):
                self.vote="1"  
            return output

        max_info = -1
        chosen_one = -1
        max_mutual_information_split = None

        for i in range(len(dict_data) - 1):
            if i in checked_attributes:
                continue

            count0, count1 = 0, 0
            count0_equal, count1_equal = 0, 0

            left_ = []
            right_ = []

            for row in train_data:
                if row[i] == "0":
                    left_.append(row)
                    count0 += 1
                    if row[-1] == "0":
                        count0_equal += 1
                elif row[i] == "1":
                    right_.append(row)
                    count1 += 1
                    if row[-1] == "0":
                        count1_equal += 1

            total = count0 + count1
            if count0 == 0:
                a = 0
            else:
                p0_equal = count0_equal / count0
                p0_not_equal = (count0 - count0_equal) / count0
                a = (count0 / total) * (-(p0_equal * np.log2(p0_equal) if p0_equal > 0 else 0) - (p0_not_equal * np.log2(p0_not_equal) if p0_not_equal > 0 else 0))

            if count1 == 0:
                b = 0
            else:
                p1_equal = count1_equal / count1
                p1_not_equal = (count1 - count1_equal) / count1
                b = (count1 / total) * (-(p1_equal * np.log2(p1_equal) if p1_equal > 0 else 0) - (p1_not_equal * np.log2(p1_not_equal) if p1_not_equal > 0 else 0))

            conditional_entropy = a + b
            info_gain = entropy - conditional_entropy

            if info_gain > max_info:
                max_info = info_gain
                chosen_one = i
                max_mutual_information_split = (left_, right_)

        if chosen_one == -1:
            labels = [row[-1] for row in train_data]
            maxi_frequency = max(set(labels), key=labels.count)
            self.vote = maxi_frequency 
            return output

        self.attr = chosen_one
        self.left = Node(self.max_depth)
        self.right = Node(self.max_depth)

        left_zeros = sum(row[-1] == "0" for row in max_mutual_information_split[0])
        left_ones = sum(row[-1] == "1" for row in max_mutual_information_split[0])
      
        right_zeros = sum(row[-1] == "0" for row in max_mutual_information_split[1])
        right_ones = sum(row[-1] == "1" for row in max_mutual_information_split[1])

        output.append("| " * (depth+1)+f"{attribute_names[self.attr]} = 0: [{left_zeros} 0/{left_ones} 1]")

        self.left.train(depth + 1, self.process(max_mutual_information_split[0]), max_mutual_information_split[0], checked_attributes + [chosen_one], output, attribute_names)
        output.append("| " * (depth+1)+f"{attribute_names[self.attr]} = 1: [{right_zeros} 0/{right_ones} 1]")

        self.right.train(depth + 1, self.process(max_mutual_information_split[1]), max_mutual_information_split[1], checked_attributes + [chosen_one], output, attribute_names)

        return output
    
    def predict(self, row):
        if self.vote is not None:
            return self.vote
        if self.attr is None:
            return self.vote
        
        attr_value = row[self.attr]
        if attr_value == '0':
            if self.left is not None:
                return self.left.predict(row)
            else:
                return self.vote  
        else: 
            if self.right is not None:
                return self.right.predict(row)
            else:
                return self.vote  

def error_train_test(tree, data):
    total = len(data)
    right_checks = 0
    pred_output = []

    for sample in data:
        predicted_label = tree.predict(sample)
        actual_label = sample[-1]
        pred_output.append(predicted_label)

        if predicted_label == actual_label:
            right_checks += 1
    accuracy = right_checks / total
    error = 1 - accuracy
 
    return error, pred_output


def final_output(train_in,test_in,max_depth,train_out,test_out,metrics_out,print_out):
    train_data = np.genfromtxt(train_in, delimiter='\t',dtype=str)

    lst=[]
    lst.extend(train_data.tolist())
    test_data = np.genfromtxt(test_in, delimiter='\t',dtype=str)
    lstest=[]
    lstest.extend(test_data.tolist())

    tree = Node(max_depth=max_depth)
    atts=tree.get_attribute_names(lst)
    o=tree.train(depth=0, dict_data=tree.process(train_data), train_data=train_data,checked_attributes=[], output=[], attribute_names=atts)

    with open(print_out, 'w') as f:
        ones=0
        zeroes=0
        for i in lst[1:]:
            if i[-1]=="1":
                ones+=1
            elif i[-1]=="0":
                zeroes+=1
        f.write("["+str(zeroes) + " 0/" +str(ones)+" 1"+"]"+'\n')
        for ele in o:
            f.write(ele + '\n')

    train_error, train_predictions = error_train_test(tree, lst[1:])
    test_error,test_predictions=error_train_test(tree, lstest[1:])
    with open(metrics_out, 'w') as f:
        f.write(f"error(train): {train_error}\n")
        f.write(f"error(test): {test_error}\n")

    with open(train_out,"w") as f:
        for t in train_predictions:
            if t!=None:
                f.write(t + '\n')
            else:
                t="0"
                f.write(t + '\n')

    with open(test_out,"w") as f:
        for t in test_predictions:
            if t!=None:
                f.write(t + '\n')
            else:
                t="0"
                f.write(t + '\n')

if __name__ == '__main__':
    train_in = sys.argv[1]
    test_in = sys.argv[2]
    max_depth=int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out= sys.argv[6]
    print_out=sys.argv[7]
    final_output(train_in,test_in,max_depth,train_out,test_out,metrics_out,print_out)


