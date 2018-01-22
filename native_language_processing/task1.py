import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_data = pd.read_csv('train1.csv', sep=",", names=["id", "text", "is_positive"], skiprows=[0], nrows=100, dtype={"is_positive": 'int'})
tdata = pd.read_csv('test1.csv', sep=",", names=["id", "text"], skiprows=[0], nrows=100, dtype={"is_positive": 'int'})
print(train_data.head())
print(tdata.head())

data = pd.DataFrame(columns=['is_positive'])
test_data = pd.DataFrame()


def process_review(review, is_positive, df):
    df = df.append(pd.DataFrame([np.zeros(len(df.columns), dtype=np.int)], columns=df.columns))
    df.iloc[-1, df.columns.get_loc("is_positive")] = is_positive
    for word in review.split():
        if (not df.__contains__(word)):
            df[word] = 0
            df.iloc[-1, df.columns.get_loc(word)] += 1
    return df


def process_reviews(df, data_to_parse):
    for index, row in data_to_parse.iterrows():
        df = process_review(row["text"], row["is_positive"], df)
    for column in df.columns:
        if sum(df[column]) < 10:
            df.drop(column)
    return df


data = process_reviews(data,train_data)
test_data = process_reviews(test_data,tdata)

print(data)

Y = results = list(map(int, data['is_positive']))
X = data.drop('is_positive')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
print(X)
print(Y)


from sklearn import discriminant_analysis
lda_model = discriminant_analysis.LinearDiscriminantAnalysis()
lda_model.fit(x_train, y_train)
err_train = np.mean(y_train != lda_model.predict(x_train))
err_test = np.mean(y_test != lda_model.predict(x_test))
print(err_train, err_test)







