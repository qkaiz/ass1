'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Chris Clingerman
'''
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score
def evaluatePerformance(num_trials=100, num_folds=10):
     # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n,d = X.shape

    # shuffle the data
    idx = np.arange(n)
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx] 

    # Initialize the array to store the accuracy for each model
    accuracies = {
        'full_tree': [],
        'stump': [],
        'three_level_tree': []
    }

     # Initialize a list to store stats for each percentage of the training data
    training_sizes = np.linspace(0.1, 0.9, 9)  # from 10% to 90%
    accuracies = {depth: {size: [] for size in training_sizes} for depth in [1, 3, None]}

    # Perform trials
    for trial in range(num_trials):
        # Shuffle data
        idx = np.arange(n)
        np.random.shuffle(idx)
        X_shuffled = X[idx]
        y_shuffled = y[idx]

        # Create folds
        fold_size = n // num_folds
        for fold in range(num_folds):
            start = fold * fold_size
            end = start + fold_size
            if fold == num_folds - 1:
                end = n

            # Split the data into training and testing sets for the current fold
            X_test = X_shuffled[start:end]
            y_test = y_shuffled[start:end]
            X_train = np.concatenate((X_shuffled[:start], X_shuffled[end:]))
            y_train = np.concatenate((y_shuffled[:start], y_shuffled[end:]))

            for train_size in training_sizes:
                subset_size = int(train_size * len(X_train))
                X_train_subset = X_train[:subset_size]
                y_train_subset = y_train[:subset_size]

                # Train and evaluate the classifiers
                for depth in accuracies.keys():
                    clf = tree.DecisionTreeClassifier(max_depth=depth)
                    clf.fit(X_train_subset, y_train_subset)
                    y_pred = clf.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    accuracies[depth][train_size].append(accuracy)

    # Calculate mean and standard deviation of accuracy for each percentage and depth
    mean_accuracies = {depth: [] for depth in accuracies}
    stddev_accuracies = {depth: [] for depth in accuracies}
    for depth, sizes in accuracies.items():
        for size, scores in sizes.items():
            mean_accuracies[depth].append(np.mean(scores))
            stddev_accuracies[depth].append(np.std(scores))

    # Calculate mean and standard deviation of accuracy for each model
    stats = np.zeros((3, 2))
    for i, model in enumerate(['full_tree', 'stump', 'three_level_tree']):
        stats[i, 0] = np.mean(accuracies[model])
        stats[i, 1] = np.std(accuracies[model])
        
    # Plot the learning curves
    plt.figure(figsize=(10, 8))
    for depth in accuracies:
        plt.errorbar(
            training_sizes * 100,  # Convert to percentages
            mean_accuracies[depth],
            yerr=stddev_accuracies[depth],
            label=f'Depth {depth}' if depth is not None else 'Unlimited Depth'
        )
    
    plt.xlabel('Percentage of Training Data Used')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves for Decision Trees of Various Depths')
    plt.legend()
    plt.grid(True)
    plt.show()

    return stats




# Do not modify from HERE...
if __name__ == "__main__":
    
    stats = evaluatePerformance()
    # print ("Decision Tree Accuracy = ", stats[0,0], " (", stats[0,1], ")")
    # print ("Decision Stump Accuracy = ", stats[1,0], " (", stats[1,1], ")")
    # print ("3-level Decision Tree = ", stats[2,0], " (", stats[2,1], ")")
# ...to HERE.
