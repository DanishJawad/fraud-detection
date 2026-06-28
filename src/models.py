from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_baseline(X_train, y_train, method='logistic'):
    if method == 'logistic':
        clf = LogisticRegression(max_iter=1000)
    else:
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', clf)
    ])
    pipe.fit(X_train, y_train)
    return pipe
