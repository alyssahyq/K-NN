
import numpy as np

"""
You need to implement this method. 
    
Input:
input_features: It will have dimensions n x d,
where n will be number of samples and d will be number of features for each sample.
num_neighbors: They are the number of neighbors that will be look while
determining the class of the input features.
true_labels: The true label of the features with dimensions n x 1, where n
is the number of samples.

    
Output:
predicted_class: It will be the list of the class labels for each sample. The 
dimension of the list will be n x 1, where n will be number of samples
f1_score:Weighted F1-Score of the prediction.
confusion_matrix = confusion matrix.
"""

def predictor(input_features,num_neighbors,true_labels):
    predicted_class_list=[]
    for i in range(input_features.shape[0]):
        test_dp = input_features[i]
        distance = []
        for j in range(input_features.shape[0]):
            this_dp = input_features[j]
            dis = np.sqrt(pow((test_dp[0]-this_dp[0]), 2) + pow((test_dp[1]-this_dp[1]), 2) + pow((test_dp[2]-this_dp[2]), 2) + pow((test_dp[3]-this_dp[3]), 2))
            distance.append(dis)
        dis_label_list = []
        dis_label_list.append(distance)
        dis_label_list.append(true_labels)
        dis_label = np.array(dis_label_list).T
        dis_label = dis_label[dis_label[:,0].argsort()]
        k_points = dis_label[1:num_neighbors+1]
        label_0 = 0
        label_1 = 0
        label_2 = 0
        for k in range(k_points.shape[0]):
            if k_points[k][1] == 0:
                label_0 = label_0 +1
            if k_points[k][1] == 1:
                label_1 = label_1 +1
            if k_points[k][1] == 2:
                label_2 = label_2 +1
        label_list = [label_0, label_1, label_2]
        max_label = label_list.index(max(label_list))
        predicted_class_list.append(max_label)
    predicted_class = np.array(predicted_class_list)
    predicted_class = predicted_class.reshape(predicted_class.shape[0], 1)
    #results[0]:TP, results[1]:FP, results[2]:FN
    #columns of results: label
    results = np.zeros((3,3),dtype=int)
    confusion_matrix = np.zeros((3,3),dtype=int)
    for x in range(true_labels.shape[0]):
        true = int(true_labels[x])
        pred = int(predicted_class[x])
        confusion_matrix[true][pred] = confusion_matrix[true][pred]+1
        if (true == pred):
            results[0][true] = results[0][true]+1
        if (true != pred):
            results[2][true] = results[2][true]+1
            results[1][pred] = results[1][pred]+1
    precision = []
    recall = []
    for l in range(results.shape[1]):
        TP = results[0][l]
        FP = results[1][l]
        FN = results[2][l]
        precision_this = TP/(TP+FP)
        recall_this = TP/(TP+FN)
        precision.append(precision_this)
        recall.append(recall_this)
    f1=[]
    for s in range(len(precision)):
        f1_this = 2 * precision[s] * recall[s] / (precision[s] + recall[s])
        f1.append(f1_this)
    class_num = len(f1)
    f1_score = pow((1/class_num*sum(f1)),2)
    return predicted_class,f1_score,confusion_matrix


def main():
    # load data set A
    data_path='DataA.csv'
    data_raw = np.genfromtxt(data_path, delimiter=',')
    data_raw = np.delete(data_raw, 0, axis=0)
    dataSetSize = data_raw.shape[0]
    data_T = data_raw.T
    # replace nan with mean
    for i in range(data_T.shape[0]):
        row = data_T[i]
        not_nan = row[row == row]
        row[np.isnan(row)] = not_nan.mean()
    # max-min normolization
    data_norm_list = []
    for i in range(data_T.shape[0] -1):
        feature = data_T[i]
        min = np.min(feature)
        max = np.max(feature)
        feature_norm = []
        for data in feature:
            data_new = (data-min)/(max-min)
            feature_norm.append(data_new)
        data_norm_list.append(feature_norm)
    true_label = data_T[4]
    input_features= np.array(data_norm_list).T
    for k in range(1,20):
        results_k = predictor(input_features,k, true_label)
        confusion_matrix_k = results_k[2]
        TP_sum = 0
        for i in range(3):
            TP_sum = TP_sum + confusion_matrix_k[i][i]
        accuracy = TP_sum/120
        print('K = ',k," , accurcay = ",round(accuracy,4)," f1_score = ",round(results_k[1],4))

if __name__ == '__main__':
    main()
