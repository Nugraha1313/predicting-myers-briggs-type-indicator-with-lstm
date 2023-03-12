import os
import time
import csv
import sys
import pickle
import numpy as np
from keras.utils import pad_sequences
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.models import load_model
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from gui import Ui_MainWindow

class window(QtWidgets.QMainWindow):
    def __init__(self):
        super(window, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('myers_logo.png'))
        # self.loadTemplate()
        # self.ui.importDatasetPushButton.clicked.connect(self.read_dataset)
        self.ui.predictButton.clicked.connect(self.app)
        self.ui.importButton.clicked.connect(self.read_dataset)
        # if(isReadDataset):
        #     self.app()


    def read_dataset(self):
        time.sleep(1)
        self.ui.datasetName.clear()
        global isReadDataSet
        isReadDataSet = False
        fname = QFileDialog.getOpenFileName(self, 'Open file', '', "CSV Files (*.csv)")
        global datasetFilePath
        datasetFilePath = fname[0]
        self.ui.datasetName.append(datasetFilePath)
        # print(datasetFilePath)
        isReadDataset = True
        # self.app()

    def app(self):
        time.sleep(1)
        self.ui.outputText.clear()
        while isReadDataSet == False and datasetFilePath == '':
            time.sleep(1)
            print('Tolong Import Dataset Terlebih Dahulu')
            self.read_dataset()
        MODELS_DIR = "models"
        DATA_DIR = "data"
        TRUMP_TWEETS_PATH = os.path.join(DATA_DIR, datasetFilePath)

        DIMENSIONS = ["IE", "NS", "FT", "PJ"]
        MODEL_BATCH_SIZE = 128
        TOP_WORDS = 2500
        MAX_POST_LENGTH = 40
        EMBEDDING_VECTOR_LENGTH = 20

        final = ""

        x_test = []
        with open(TRUMP_TWEETS_PATH, "r", encoding="ISO-8859-1") as f:
            reader = csv.reader(f)
            for row in f:
                x_test.append(row)

        types = [
            "INFJ",
            "ENTP",
            "INTP",
            "INTJ",
            "ENTJ",
            "ENFJ",
            "INFP",
            "ENFP",
            "ISFP",
            "ISTP",
            "ISFJ",
            "ISTJ",
            "ESTP",
            "ESFP",
            "ESTJ",
            "ESFJ",
        ]
        types = [x.lower() for x in types]
        lemmatizer = WordNetLemmatizer()
        stop_words = stopwords.words("english")

        def lemmatize(x):
            lemmatized = []
            for post in x:
                temp = post.lower()
                for type_ in types:
                    temp = temp.replace(" " + type_, "")
                temp = " ".join(
                    [
                        lemmatizer.lemmatize(word)
                        for word in temp.split(" ")
                        if (word not in stop_words)
                    ]
                )
                lemmatized.append(temp)
            return np.array(lemmatized)

        for k in range(len(DIMENSIONS)):
            model = load_model(
                os.path.join(MODELS_DIR, "rnn_model_{}.h5".format(DIMENSIONS[k]))
            )
            tokenizer = None
            with open(
                    os.path.join(MODELS_DIR, "rnn_tokenizer_{}.pkl".format(DIMENSIONS[k])), "rb"
            ) as f:
                tokenizer = pickle.load(f)

            def preprocess(x):
                lemmatized = lemmatize(x)
                tokenized = tokenizer.texts_to_sequences(lemmatized)
                return pad_sequences(tokenized, maxlen=MAX_POST_LENGTH)

            predictions = model.predict(preprocess(x_test))
            prediction = float(sum(predictions) / len(predictions))
            # print(DIMENSIONS[k])
            # print(prediction)
            # self.ui.outputText.append(str(DIMENSIONS[k]))
            # self.ui.outputText.append(str(prediction))
            if prediction >= 0.5:
                self.ui.outputText.append(str(DIMENSIONS[k]) + " -> " + str(DIMENSIONS[k][1]))
                final += DIMENSIONS[k][1]
            else:
                self.ui.outputText.append(str(DIMENSIONS[k]) + " -> " + str(DIMENSIONS[k][0]))
                final += DIMENSIONS[k][0]

            self.ui.outputText.append(str(prediction))

        # print("")
        # print("Final Prediction: {}".format(final))
        output = "Final Prediction: {}".format(final)
        self.ui.outputText.append(output)

def app():
    app = QtWidgets.QApplication(sys.argv)
    win = window()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    app()