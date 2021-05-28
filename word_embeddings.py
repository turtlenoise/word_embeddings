# https://deepscopy.com/Create_your_Mini_Word_Embedding_from_Scratch_using_Pytorch
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch import nn
import torch
from torchsummary import summary
from matplotlib import pyplot as plt
import random
import plotly.graph_objects as go
import pandas as pd
import plotly
import regex as re

WORD_LENGTH = 1001


def store_words(words):
    f = open("saved/word_dict.txt", "w")
    for i in range(0, len(words)):
        f.write(str(i) + "," + words[i] + "\n")
    f.close()


def store_predictions(prediction, model, device):

    f = open("saved/predictions.txt", "w")
    for i in range(prediction.shape[0]):
        result, _ = model(
            torch.from_numpy(prediction[i]).unsqueeze(0).float().to(device)
        )
        resultNumpy = result.detach().numpy()
        resultNoLineBreaks = (
            resultNumpy.tolist()
        )  # np.array_repr(resultNumpy).replace('\n', '')
        f.write(str(i) + "," + str(resultNoLineBreaks) + "\n")
    f.close()


# builds two dictionaries: a word to unique numerical ID
# and another where the id is key and the word is value.
def word_indexer(corpus, words):
    idx_2_word = {}
    word_2_idx = {}
    temp = []
    i = 1
    for sentence in corpus:
        for word in sentence.split():
            if (word not in temp) and (word in words):
                temp.append(word)
                idx_2_word[i] = word
                word_2_idx[word] = i
                i += 1
    return idx_2_word, word_2_idx


def one_hot_map(doc, word_2_idx, words):
    x = []
    for word in doc.split():
        if word in words:
            x.append(word_2_idx[word])
    return x


def build_input_target_pairs(padded_docs):
    training_data = np.empty((0, 2))

    window = 2  # how many neighbours to take into consideration per word.
    for sentence in padded_docs:
        sent_len = len(sentence)
        for i, word in enumerate(sentence):
            w_context = []
            if sentence[i] != 0:
                w_target = sentence[i]
                for j in range(i - window, i + window + 1):
                    if j != i and j <= sent_len - 1 and j >= 0 and sentence[j] != 0:
                        w_context = sentence[j]
                        training_data = np.append(
                            training_data, [[w_target, w_context]], axis=0
                        )
                        # training_data.append([w_target, w_context])
    return training_data


def perform_one_hot_encoding(training_data):
    enc = OneHotEncoder()
    enc.fit(np.array(range(WORD_LENGTH)).reshape(-1, 1))
    onehot_label_x = enc.transform(training_data[:, 0].reshape(-1, 1)).toarray()

    enc = OneHotEncoder()
    enc.fit(np.array(range(WORD_LENGTH)).reshape(-1, 1))
    onehot_label_y = enc.transform(training_data[:, 1].reshape(-1, 1)).toarray()

    return onehot_label_x, onehot_label_y


class WEMB(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, input_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data_input):
        out_bn = self.layer1(data_input)
        output_layer1 = self.relu(out_bn)
        output_layer2 = self.layer2(output_layer1)
        output_layer3 = self.softmax(output_layer2)
        return output_layer3, out_bn


def train_the_model(onehot_label_x, onehot_label_y):
    input_size = WORD_LENGTH
    hidden_size = 32
    learning_rate = 0.01
    num_epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WEMB(input_size, hidden_size).to(device)
    model.train(True)
    # print(model)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0, weight_decay=0, nesterov=False
    )
    summary(model, torch.ones((1, WORD_LENGTH)))

    loss_val = []
    onehot_label_x = onehot_label_x.to(device)
    onehot_label_y = onehot_label_y.to(device)

    for epoch in range(num_epochs):
        for i in range(onehot_label_y.shape[0]):
            inputs = onehot_label_x[i].float()
            labels = onehot_label_y[i].float()
            inputs = inputs.unsqueeze(0)
            labels = labels.unsqueeze(0)

            # Forward pass
            output, wemb = model(inputs)
            loss = criterion(output, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_val.append(loss.item())

        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    return model, device, loss_val


def main():

    #### preprocessing
    filename = "./text/100-0.txt"
    with open(filename) as f:
        text = f.read()

    text = re.sub(r"(M\w{1,2})\.", r"\1", text)
    text = re.sub(r"(\n)", r"", text)
    text = re.sub("[\,\_,\],\[]", "", text)
    corpus = re.split(r' *[\.\?!][\'"\)\]]* *', text.lower())
    # print(corpus)

    wordFrequency = {}

    for sentence in corpus:
        for word in sentence.split():
            if word in wordFrequency:
                wordFrequency[word] = wordFrequency[word] + 1
            else:
                wordFrequency[word] = 1

    sortedWordFrequency = dict(sorted(wordFrequency.items(), key=lambda item: item[1]))

    # print(sortedWordFrequency)

    sortedWordFrequencyList = list(sortedWordFrequency.keys())
    # print(sortedWordFrequencyList)
    # TODO: without the most frequent 100
    i = 0
    while i < 100:
        sortedWordFrequencyList.pop()
        i = i + 1
    # print(sortedWordFrequencyList)
    # TODO: add subscript, only the 10000 most frequent after the stopwords.
    words = sortedWordFrequencyList[-(WORD_LENGTH - 1) :]
    store_words(words)
    print(words)
    print(len(words))

    if len(words) + 1 != WORD_LENGTH:
        print("set correctly the WORD_LENGTH variable.")
        print("it should be " + str((len(words) + 1)))
        exit()

    idx_2_word, word_2_idx = word_indexer(corpus, words)
    encoded_docs = [one_hot_map(d, word_2_idx, words) for d in corpus]
    max_len = WORD_LENGTH - 1
    padded_docs = pad_sequences(encoded_docs, maxlen=max_len, padding="post")
    training_data = build_input_target_pairs(padded_docs)

    onehot_label_x, onehot_label_y = perform_one_hot_encoding(training_data)

    #### hyper-parameter selection

    onehot_label_x = torch.from_numpy(onehot_label_x)
    onehot_label_y = torch.from_numpy(onehot_label_y)

    #### training the model.
    print("the learning has started.")
    model, device, loss_val = train_the_model(onehot_label_x, onehot_label_y)

    #### testing the model.
    docs = words
    test_list = []
    for i in range(1, WORD_LENGTH):
        test_list.append(i)

    test_arr = np.array(test_list)

    enc = OneHotEncoder()
    enc.fit(np.array(range(WORD_LENGTH)).reshape(-1, 1))
    test = enc.transform(test_arr.reshape(-1, 1)).toarray()

    store_predictions(test, model, device)

    output = []
    for i in range(test.shape[0]):
        result, wemb2 = model(torch.from_numpy(test[i]).unsqueeze(0).float().to(device))
        wemb2 = wemb2[0].detach().cpu().numpy()
        output.append(wemb2)
        # print("for: " + words[i])
        # print("the closest word is: ")
        resultNumpy = result.detach().numpy()
        maxIndex = np.argmax(resultNumpy, axis=1)[0]
        # print(words[maxIndex-1])

    xs = []
    ys = []
    for i in range(len(output)):
        xs.append(output[i][0])
        ys.append(output[i][1])
    print(xs, ys)

    label = docs

    fig = go.Figure(
        data=go.Scatter(
            x=xs,
            y=ys,
            text=label,
            mode="markers",
            marker=dict(
                size=16,
                color=np.random.randn(500),  # set color equal to a variable
                colorscale="Viridis",  # one of plotly colorscales
                showscale=True,
            ),
        )
    )  # hover text goes here

    fig.update_layout(title="word embeddings")
    plotly.offline.plot(fig, filename="word_embedding_results.html")
    fig.show()


if __name__ == "__main__":
    main()
