from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
diabetes = load_diabetes()
train_x, train_y = diabetes.data[:400], diabetes.target[:400]
test_x, test_y = diabetes.data[400:], diabetes.target[400:]

def models(n_neighbors=5, alpha=1.0):
    base_learners = []
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    base_learners.append(knn)
    dtr = DecisionTreeRegressor(max_depth=4 , random_state=123456)
    base_learners.append(dtr)
    ridge = Ridge(alpha=alpha)
    base_learners.append(ridge)
    meta_learner = LinearRegression()
    return base_learners, meta_learner

def section2(base_learners):
    meta_data = np.zeros((len(base_learners), len(train_x)))
    meta_targets = np.zeros(len(train_x))
    KF = KFold(n_splits=5)
    meta_index = 0
    for train_indices, test_indices in KF.split(train_x):
        for i in range(len(base_learners)):
            learner = base_learners[i]
            learner.fit(train_x[train_indices], train_y[train_indices])
            predictions = learner.predict(train_x[test_indices])
            meta_data[i][meta_index:meta_index+len(test_indices)] = predictions
        meta_targets[meta_index:meta_index+len(test_indices)] = train_y[test_indices]
        meta_index += len(test_indices)
    meta_data = meta_data.transpose()
    return meta_data, meta_targets

def section3(base_learners):
    test_meta_data = np.zeros((len(base_learners), len(test_x)))
    base_errors = []
    base_r2 = []
    for i in range(len(base_learners)):
        learner = base_learners[i]
        learner.fit(train_x, train_y)
        predictions = learner.predict(test_x)
        test_meta_data[i] = predictions
        err = metrics.mean_squared_error(test_y, predictions)
        r2 = metrics.r2_score(test_y, predictions)
        base_errors.append(err)
        base_r2.append(r2)
    test_meta_data = test_meta_data.transpose()
    return test_meta_data, base_errors, base_r2

def section4(meta_learner, meta_data, meta_targets, test_meta_data):
    meta_learner.fit(meta_data, meta_targets)
    ensemble_predictions = meta_learner.predict(test_meta_data)
    err = metrics.mean_squared_error(test_y, ensemble_predictions)
    r2 = metrics.r2_score(test_y, ensemble_predictions)
    return err, r2

def print_model(base_learners, base_errors, base_r2, err, r2):
    print('ERROR   R2    Name')
    print('-'*20)
    for i in range(len(base_learners)):
      learner = base_learners[i]
      print(f'{base_errors[i]:.1f} {base_r2[i]:.2f} {learner.__class__.__name__}')
      print(f'{err:.1f} {r2:.2f} Ensemble\n')
    
def main():
    model_base,model_meta  = models()
    meta_x, meta_y= section2(model_base)
    test_metadata, be, br2 = section3(model_base)
    e, R2 = section4(model_meta, meta_x, meta_y, test_metadata)
    print_model(model_base, be, br2, e, R2)
    
if __name__ == '__main__': 
    main()