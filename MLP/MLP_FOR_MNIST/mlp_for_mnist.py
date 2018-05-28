from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
cls = MLPClassifier(activation="relu", alpha=1e-05, batch_size='auto', beta_1=0.9,
                    beta_2=0.999, early_stopping=False, epsilon=1e-08,
                    hidden_layer_sizes=(100, 100), learning_rate='constant',
                    learning_rate_init=0.001, max_iter=200, momentum=0.9,
                    nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                    solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
                    warm_start=False)
print('accuracy： %s' % cross_val_score(cls, X, y, cv=5).mean())
