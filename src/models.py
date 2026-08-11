from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


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


def train_smote_pipeline(X_train, y_train, random_state=42):
    """Train StandardScaler -> SMOTE -> RandomForest as one imblearn Pipeline.

    imblearn's Pipeline only resamples during .fit(), never at .predict()/
    .predict_proba() time, so SMOTE structurally cannot leak into validation
    or test data even under cross-validation.
    """
    pipe = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=random_state)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)
    return pipe
