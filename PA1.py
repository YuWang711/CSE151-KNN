import numpy as np
import math
import random

def euclidean_distance(vector1, vector2):
    # result_sum = 0
    # for i in range(len(vector1)):
    #     result_sum += ((vector1[i] - vector2[i])**2)
    # result_sum = math.sqrt(result_sum)
    vector1_np = np.array(vector1)
    vector2_np = np.array(vector2)
    return (np.linalg.norm(vector1_np - vector2_np)**2)

def readFile(filename):
    data_array = []
    #Read the file name that was passed in as paramter
    with open(filename,'r') as testing:
        #for each line store the array as pair [[0:783],784]
        for eachline in testing:
            split_line = eachline.split()
            data = []
            #get the first 784 index
            for i in range(len(split_line)-1):
                data.append(float(split_line[i]))
            #get the label
            label = float(split_line[len(split_line)-1])
            #add the pair into array
            data_array.append([data,label])
    return data_array

def find_index_of_max(digits_array):
    #find the digits that occur the most per data
    index_of_max = 0
    max_index = []
    for i in range(len(digits_array)):
        #if index has more count, change the max index
        if(digits_array[int(index_of_max)] < digits_array[int(i)]):
            index_of_max = i
            max_index = [i]
        #if index has as much count as the max index, pick it randomly
        elif(digits_array[int(index_of_max)] == digits_array[int(i)]):
            max_index.append(i)
    index_of_max = random.choice(max_index)
    return int(index_of_max)

def k_nearest_neighbor(k,training, validation):
    #KNN
    validation_correct = 0
    #Training is the training set
    #validation is the set we are comparing our training set to.
    for pair in validation:
        e_distance = []
        #digits is a array size of 10 that counts
        digits = np.zeros(10)
        #for each training pair find euclidean distance
        for training_pair in training:
            e_distance.append([euclidean_distance(pair[0],training_pair[0]), training_pair[1]])
        e_distance.sort()
        #Count the occurrence of the label nearest to validation data
        for i in range(k):
            digits[int(e_distance[i][1])] += 1
        prediction = find_index_of_max(digits)
        #Counting the prediction that got correct.
        if(prediction == pair[1]):
            validation_correct += 1
    print("Error for {}-nn is {}".format(k,1-(validation_correct/len(validation))) )
    return (validation_correct/len(validation))

def find_best_K(training,validation,testing):
    #Use the follow k for k-nn 
    Ks = [1,3,5,9,15]
    validation_result = []
    #run K-nn for training vs validation
    print("Validation: ")
    for k in Ks:
        validation_result.append(k_nearest_neighbor(k, training, validation))
    index = find_index_of_max(validation_result)
    print("{}-NN has the smallest validation error. ".format(Ks[index]))

    #run K-nn for training vs training
    print("Trainning: ")
    trainning_result = []
    for k in Ks:
        trainning_result.append(k_nearest_neighbor(k, training, training))
    
    #run K-nn for training vs testing 
    print("Testing: ")
    testing_result = []
    testing_result.append(k_nearest_neighbor(Ks[index], training, testing))
    return Ks[index]

def find_projected(image):
    projection_matrix = []
    #read the projection.txt
    with open('projection.txt','r') as projection:
        for eachline in projection:
            split_line = eachline.split()
            data = []
            for i in range(len(split_line)):
                data.append(float(split_line[i]))
            projection_matrix.append(data)
    #Store it in the format of [784x20]
    projection_matrix = np.array(projection_matrix)
    project_matrix = []
    #reshape each input matrix to 1x784
    for vector in image:
        temp_vector = np.array(vector[0])
        np.reshape(temp_vector,(1,len(temp_vector)))
        #dot product of [1 x 784] * [ 784 x 20 ]
        project_matrix.append([np.dot(temp_vector,projection_matrix),vector[1]])
    return project_matrix


def main():
    train_set = readFile('pa1Train.txt')
    train_set = np.array(train_set)
    test_set = readFile('pa1Testing.txt')
    test_set = np.array(test_set)
    validation_set = readFile('pa1Validate.txt')
    validation_set = np.array(validation_set)

    projected_train_set = find_projected(train_set)
    projected_train_set = np.array(projected_train_set)
    projected_validation_set = find_projected(validation_set)
    projected_validation_set = np.array(projected_validation_set)
    projected_test_set = find_projected(test_set)
    projected_test_set = np.array(projected_test_set)
    n = find_best_K(train_set, validation_set,test_set)
    find_best_K(projected_train_set,projected_validation_set,projected_test_set)
    
    return


if __name__== "__main__" :
    main()