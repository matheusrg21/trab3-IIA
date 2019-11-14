import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load in data
df = pd.read_csv('selfie_dataset.txt', names=['img', 'score', 'partial_faces', 'is_female', 'baby', 'child', 'teenager', 
                                                'youth', 'middle_age', 'senior', 'white', 'black', 'asian', 'oval_face', 
                                                'round_face', 'heart_face', 'smiling', 'mouth_open', 'frowning', 
                                                'wearing_glasses', 'wearing_sunglasses', 'wearing_lipstick',
                                                'tongue_out', 'duck_face', 'black_hair', 'blond_hair', 'brown_hair', 
                                                'red_hair', 'curly_hair', 'straight_hair', 'braid_hair', 
                                                'showing_cellphone', 'using_earphone', 'using_mirror', 'braces', 
                                                'wearing_hat', 'harsh_lighting', 'dim_lighting'
                                                ], 
                                                usecols=lambda column: column not in ["img", "score"], sep=' ')


columns = df.columns.tolist()

print(columns)



train = df.sample(frac=0.7, random_state=1)

test = df.loc[~df.index.isin(train.index)]

print(train.shape)

print(test.shape)

model = RandomForestRegressor(n_estimators=100, min_samples_leaf=8, random_state=1)

# Fit the model to the data.
model.fit(train[columns], train)
# Make predictions.
predictions_rf = model.predict(test[columns])
# Compute the error.
# mean_squared_error(predictions_rf, test)

features=df.columns
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()

