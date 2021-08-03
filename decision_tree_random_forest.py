def split(dataset, labels, column):
    data_subsets = []
    label_subsets = []
    # print([data[column] for data in dataset])
    counts = list(set([data[column] for data in dataset]))
    # print(counts)
    counts.sort()
    # print(counts)
    # print(len(dataset))
    for k in counts:
        new_data_subset = []
        new_label_subset = []
        for i in range(len(dataset)):
            if dataset[i][column] == k:
                new_data_subset.append(dataset[i])
                new_label_subset.append(labels[i])
        data_subsets.append(new_data_subset)
        label_subsets.append(new_label_subset)
    return data_subsets, label_subsets

def gini(dataset):
  impurity = 1
  label_counts = Counter(dataset)
  for label in label_counts:
    prob_of_label = label_counts[label] / len(dataset)
    impurity -= prob_of_label ** 2
  return impurity

def information_gain(starting_labels, split_labels):
  info_gain = gini(starting_labels) 
  for subset in split_labels:
    # Multiply gini(subset) by the correct percentage below
    info_gain -= gini(subset) * (len(subset) / len(starting_labels))
  return info_gain

  
def find_best_split(dataset, labels):
    features = np.random.choice(len(dataset[0]), 3, replace=False)
    
    best_gain = 0
    best_feature = 0
#     for feature in range(len(dataset[0])):
      for feature in range(features):
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_feature, best_gain

def build_tree(data, labels):
  best_feature, best_gain = find_best_split(data, labels)
  if best_gain == 0:
    return Counter(labels)
  data_subsets, label_subsets = split(data, labels, best_feature)
  branches = []
  for i in range(len(data_subsets)):
    if i == 0:
      print(data_subsets[i])
    branch = build_tree(data_subsets[i], label_subsets[i])
    branches.append(branch)
  return branches
  
tree = build_tree(car_data, car_labels)
print_tree(tree)  

  
def classify(datapoint, tree):
  if isinstance(tree, Leaf):
    return max(tree.labels.items(), key=operator.itemgetter(1))[0]

  value = datapoint[tree.feature]
  for branch in tree.branches:
    if branch.value == value:
      return classify(datapoint, branch)
    
indices = [random.randint(0, 999) for i in range(1000)]

    
tree = make_single_tree(training_data, training_labels)
single_tree_correct = 0
forest = make_random_forest(40, training_data, training_labels)
forest_correct = 0
predictions = []

for i in range(20):


  data_subset = [car_data[index] for index in indices]
  labels_subset = [car_labels[index] for index in indices]
  subset_tree = build_tree(data_subset, labels_subset)
  print(classify(unlabeled_point, subset_tree))
  predictions.append(classify(unlabeled_point, subset_tree))
