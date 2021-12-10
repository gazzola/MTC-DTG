# -*- coding: utf-8 -*-
""" Single Task Learning (STL) for books
This file represents the implementation of the Single Task Learning used in the research.
Single Task Learning (STL) approach and two Multi-Task Learning (MTL) implementations with two tasks,
called Multi-task approach to Text Complexity using Different Text Genres (MTC-DTG) Simplex,
and with three tasks, called Multi-task approach to Text Complexity using Different Text Genres (MTC-DTG)
In this work, we assessed the effect of class balancing with an oversampling and undersampling approach.
We have also used data augmentation techniques, which were necessary for MTL architectures to have equal data entry.

The files used, such as books_only_metrics_run.txt, dataset1_dataset2.csv, books.txt, metrics_3.txt, among others. These are metrics extracted from the corpora using NILC Metrics.

Example:
        To run this experiment, you must extract the metrics, if you haven't already,
        and insert in a CSV and use the names of the features presented in
        def load_livros_single. The list of features needed as feats.

        $ python single_task_book.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Todo:
    * For module TODOs
    *

.. _NILC-ICMC USP Group - Other tools:
   http://www.nilc.icmc.usp.br/nilc/index.php/tools-and-resources

"""
import tensorflow.keras
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import argmax
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from tensorflow.keras.layers import Dense, Dropout, Activation
import pathlib
import tensorflow as tf
import os


def load_livros_single():
    """This function is to load the extracted book metrics already stored.
    It works as a data loading function for training, since we can change
    the data loading to other sources. Some use generic data loading functions.
    However, as we use a loading with oversampling and more transformation
    functions specific to books, we prefer to leave a loading itself and
    not inline or generic.

    Args:
        None

    Returns:
        Returns the metrics data (X) and their labels (Y) inside a pandas object.

    """
    df = pandas.read_csv("books_only_metrics_run.txt", delimiter=',', header=0)
    # 79 features
    feats = ['adjective_ratio', 'adverbs', 'syllables_per_content_word', 'words_per_sentence', 'noun_ratio',
             'pronoun_ratio', 'verbs', 'negation_ratio', 'cw_freq', 'min_cw_freq', 'first_person_pronouns', 'ttr',
             'conn_ratio', 'add_neg_conn_ratio', 'add_pos_conn_ratio', 'cau_neg_conn_ratio', 'cau_pos_conn_ratio',
             'log_neg_conn_ratio', 'log_pos_conn_ratio', 'tmp_neg_conn_ratio', 'tmp_pos_conn_ratio',
             'adjectives_ambiguity', 'adverbs_ambiguity', 'nouns_ambiguity', 'verbs_ambiguity', 'yngve', 'frazier',
             'dep_distance', 'words_before_main_verb', 'mean_noun_phrase', 'min_noun_phrase', 'max_noun_phrase',
             'std_noun_phrase', 'passive_ratio', 'adj_arg_ovl', 'arg_ovl', 'adj_stem_ovl', 'stem_ovl', 'adj_cw_ovl',
             'third_person_pronouns', 'concretude_mean', 'concretude_std', 'concretude_1_25_ratio',
             'concretude_25_4_ratio', 'concretude_4_55_ratio', 'concretude_55_7_ratio', 'content_word_diversity',
             'familiaridade_mean', 'familiaridade_std', 'familiaridade_1_25_ratio', 'familiaridade_25_4_ratio',
             'familiaridade_4_55_ratio', 'familiaridade_55_7_ratio', 'idade_aquisicao_mean', 'idade_aquisicao_std',
             'idade_aquisicao_1_25_ratio', 'idade_aquisicao_4_55_ratio', 'idade_aquisicao_55_7_ratio',
             'idade_aquisicao_25_4_ratio', 'imageabilidade_mean', 'imageabilidade_std', 'imageabilidade_1_25_ratio',
             'imageabilidade_25_4_ratio', 'imageabilidade_4_55_ratio', 'imageabilidade_55_7_ratio',
             'sentence_length_max', 'sentence_length_min', 'sentence_length_standard_deviation', 'verb_diversity',
             'adj_mean', 'adj_std', 'all_mean', 'all_std', 'givenness_mean', 'givenness_std', 'span_mean', 'span_std',
             'content_density', 'ratio_function_to_content_words']

    X = df[feats].values
    Y = df[['classe']]

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(Y.values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    Y = onehot_encoded

    scaler = StandardScaler()
    scaler.fit(X)

    X = scaler.transform(X)

    ## Oversampler
    os = RandomOverSampler(sampling_strategy='all')
    X_new, Y_new = os.fit_sample(X, Y)

    df_book_final = pd.DataFrame(X_new, columns=feats)
    X_new_df = df_book_final[feats]

    ### Returns list of class ###
    df_2 = pandas.read_csv("books.txt", delimiter=',', header=0)

    Y_2 = df_2[['classe']]

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(Y_2.values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    Y_2 = onehot_encoded

    decode_label = []
    for i in range(len(Y_new)):
        decode_label.append(label_encoder.inverse_transform([argmax(Y_new[i, :])]))

    import numpy
    unique, counts = numpy.unique([decode_label], return_counts=True)
    lista_classes_count = dict(zip(unique, counts))
    lista_classes_count

    ###

    print(X_new.shape, Y_new.shape)
    return X_new_df, Y_new

def baseline_model_79feat_book():
    """This function creates an architecture for the model based on Single Task Learning,
    and uses the relu and softmax activation functions, and the categorical_crossentropy
    loss function.

        Args:
            None

        Returns:
            Returns the model based on a pre-built architecture for training as an object of keras 2.0

        """

    model = Sequential()  # 100, 50
    model.add(Dense(120, input_dim=79, activation='relu', kernel_initializer='random_normal'))
    model.add(Dense(28, activation='relu', kernel_initializer='random_normal'))
    model.add(Dense(14, activation='relu', kernel_initializer='random_normal'))
    model.add(Dense(4, activation='softmax', kernel_initializer='random_normal'))
    model.compile(loss='categorical_crossentropy', optimizer=tensorflow.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model

model_dir=pathlib.Path().absolute()
checkpoint_filepath = os.path.join(model_dir, 'save_model')
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='accuracy',
    mode='max',
    save_best_only=True)

X_new_df, Y_new = load_livros_single()
pipeline = baseline_model_79feat_book()

total_acuracia = 0
total_fscore = 0
n_split = 10
bff_fscore=0.0
for train_index, test_index in KFold(n_split).split(X_new_df):
    X_train_pss, Y_train_pss = X_new_df.iloc[train_index], Y_new[train_index]
    X_test_pss, Y_test_pss = X_new_df.iloc[test_index], Y_new[test_index]

    pipeline.fit(X_train_pss, Y_train_pss, epochs=30, batch_size=10, verbose=0, callbacks=[model_checkpoint_callback])
    prediction = pipeline.predict(X_test_pss)

    Y_test_pss_class = []
    for i in Y_test_pss:
        Y_test_pss_class.append(np.argmax(i))

    prediction_class = []
    for j in range(0, len(prediction)):
        prediction_class.append(np.argmax(prediction[j]))

    print("Accuracy:", accuracy_score(Y_test_pss_class, prediction_class))

    total_acuracia += accuracy_score(Y_test_pss_class, prediction_class)
    precision, recall, fscore, support = score(Y_test_pss_class, prediction_class, average='weighted')
    if (fscore>bff_fscore):
        bff_fscore=fscore

    print("---------------")
    print("Recall %s" % recall)
    print("Precision %s" % precision)
    print("Fscore %s" % fscore)
    total_fscore += fscore
    print("---------------")

print("=================")
print("Total Accuracy %s" % (total_acuracia / n_split))
print("Total fscore (weighted) %s" % (total_fscore / n_split))
print("=================")


