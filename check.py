import base64
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics


def make_data_from_words(file_name):
    with open(file_name) as f:
        data = f.read().splitlines()
        length_of_data = len(data)
        half_of_array = length_of_data//2
        random.shuffle(data)
        for d in range(half_of_array):
            data[d] = (str(base64.b64encode(bytes(data[d], 'utf-8'))), 1)
        for d in range(half_of_array, length_of_data):
            data[d] = (data[d], 0)
        random.shuffle(data)
        return data


def process_string(string):
    length_of_string = len(string)
    has_equal_sign = 0
    has_dash_sign = 0
    has_number = 0
    if '=' in string:
        has_equal_sign = 1
    if '\'' in string:
        has_dash_sign = 1
    if [s for s in string if s in '1234567890']:
        has_number = 1
    return [length_of_string, has_equal_sign, has_dash_sign, has_number]


def form_x_and_y(data):
    x = []
    y = []
    for d in data:
        features = process_string(d[0])
        x.append(features)
        y.append(d[1])
    return x, y


def fit(x, y, classifier):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    classifier.fit(x_train, y_train)
    print('Accuracy of Logistic regression classifier on training set: {:.2f}'
          .format(classifier.score(x_train, y_train)))
    print('Accuracy of Logistic regression classifier on test set: {:.2f} \n'
          .format(classifier.score(x_test, y_test)))

    y_pred_proba = classifier.predict_proba(x_test)[::,1]
    build_roc_curve(y_test, y_pred_proba)


def predict(string, classifier):
    processed_string = [process_string(string)]
    prediction = classifier.predict(processed_string)[0]
    return "string" if prediction == 0 else "base64"


def add_base64_to_text(text):
    copy_of_text = []
    for word in text:
        r = random.randint(0, 1)
        if r == 0:
            word = str(base64.b64encode(bytes(word, 'utf-8')))
            copy_of_text.append(word)
            continue
        copy_of_text.append(word)
    return copy_of_text


def process_text(file_name, classifier):
    with open(file_name) as f:
        text = f.read().split()
        text_with_base64 = add_base64_to_text(text)
        for word in text_with_base64:
            print(word + " : " + predict(word, classifier))


def build_roc_curve(y_test, y_pred_proba):
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.show()


def main():
    data = make_data_from_words("data.txt")
    x, y = form_x_and_y(data)
    classifier = LogisticRegression()
    fit(x, y, classifier)
    process_text("text2.txt", classifier)


if __name__ == '__main__':
    main()
