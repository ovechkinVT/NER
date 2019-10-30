# python 3

import os
import pandas as pd
import csv

from tqdm import tqdm


def get_sent_and_tags_from_tsv(full_file_name):
    """ Функция преобразует разметфу из формата tsv в формат [sentence: list, tags: list].
    """

    df = pd.read_csv(full_file_name, sep="\t",
                     header=None, quoting=csv.QUOTE_NONE)
    df = df.loc[:, :1]
    df.columns = ["word", "tag"]

    split_ix = df.index[(df["word"] == ".") & (df["tag"] == "O")].tolist()  # sent split index
    words = df["word"].tolist()
    tags = df["tag"].tolist()

    sent = [words[start + 1:end + 1] for start, end in zip([0] + split_ix, split_ix)]
    tags = [tags[start + 1:end + 1] for start, end in zip([0] + split_ix, split_ix)]

    return list(zip(sent, tags))

def load_data_from(path):
    """ Функция выгружает все файыл из директории path и преобразует их к нужному виду."""

    books = sorted(os.listdir(path))

    books_train = books[0:80]
    books_dev = books[80:90]
    books_test = books[90:100]

    sent_train = list()
    for book in tqdm(books_train):
        sent_train.extend(get_sent_and_tags_from_tsv(path + book))

    sent_dev = list()
    for book in tqdm(books_dev):
        sent_dev.extend(get_sent_and_tags_from_tsv(path + book))

    sent_test = list()
    for book in tqdm(books_test):
        sent_test.extend(get_sent_and_tags_from_tsv(path + book))

    return sent_train, sent_dev, sent_test


if __name__ == "__main__":

    path = "litbank/entities/tsv/"
    sent_train, sent_dev, sent_test = load_data_from(path)
    print(len(sent_train), len(sent_dev), len(sent_test))