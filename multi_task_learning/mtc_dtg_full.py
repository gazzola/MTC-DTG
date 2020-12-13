# -*- coding: utf-8 -*-
""" MTC-DTG
This file represents the implementation of the Multi-Task Learning (MTL) implementations with two tasks,
called Multi-task approach to Text Complexity using Different Text Genres (MTC-DTG) Simplex,
and with three tasks, called Multi-task approach to Text Complexity using Different Text Genres (MTC-DTG)
In this work, we assessed the effect of class balancing with an oversampling and undersampling approach.
We have also used data augmentation techniques, which were necessary for MTL architectures to have equal data entry.

Example:
        To run this experiment, you must extract the metrics, if you haven't already,
        and insert in a CSV and use the names of the features presented in
        def load_livros_single. The list of features needed as feats.

        $ python mtc_dtg_full.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Todo:
    * For module TODOs
    *

.. _NILC-ICMC USP Group - Other tools:
   http://www.nilc.icmc.usp.br/nilc/index.php/tools-and-resources

"""


#Library

import pandas
import tensorflow.keras
from tensorflow.keras.layers import Dense, Activation
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from numpy import argmax
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from sklearn.model_selection import KFold
from keras import Model, layers
from sklearn.metrics import accuracy_score
from keras.layers import Dense, Input
from sklearn.metrics import precision_recall_fscore_support as score
from keras.layers import Dense, Dropout, Activation
import numpy as np

def load_dataset1_dataset2():
    """This function load_dataset1_dataset2 is to load the extracted metrics already stored.
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
    df_livros = pandas.read_csv("dataset1_dataset2.csv", delimiter=',', header=0)
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
             'content_density', 'ratio_function_to_content_words', 'classe']
    feats_books = ['adjective_ratio', 'adverbs', 'syllables_per_content_word', 'words_per_sentence', 'noun_ratio',
                   'pronoun_ratio', 'verbs', 'negation_ratio', 'cw_freq', 'min_cw_freq', 'first_person_pronouns', 'ttr',
                   'conn_ratio', 'add_neg_conn_ratio', 'add_pos_conn_ratio', 'cau_neg_conn_ratio', 'cau_pos_conn_ratio',
                   'log_neg_conn_ratio', 'log_pos_conn_ratio', 'tmp_neg_conn_ratio', 'tmp_pos_conn_ratio',
                   'adjectives_ambiguity', 'adverbs_ambiguity', 'nouns_ambiguity', 'verbs_ambiguity', 'yngve',
                   'frazier', 'dep_distance', 'words_before_main_verb', 'mean_noun_phrase', 'min_noun_phrase',
                   'max_noun_phrase', 'std_noun_phrase', 'passive_ratio', 'adj_arg_ovl', 'arg_ovl', 'adj_stem_ovl',
                   'stem_ovl', 'adj_cw_ovl', 'third_person_pronouns', 'concretude_mean', 'concretude_std',
                   'concretude_1_25_ratio', 'concretude_25_4_ratio', 'concretude_4_55_ratio', 'concretude_55_7_ratio',
                   'content_word_diversity', 'familiaridade_mean', 'familiaridade_std', 'familiaridade_1_25_ratio',
                   'familiaridade_25_4_ratio', 'familiaridade_4_55_ratio', 'familiaridade_55_7_ratio',
                   'idade_aquisicao_mean', 'idade_aquisicao_std', 'idade_aquisicao_1_25_ratio',
                   'idade_aquisicao_4_55_ratio', 'idade_aquisicao_55_7_ratio', 'idade_aquisicao_25_4_ratio',
                   'imageabilidade_mean', 'imageabilidade_std', 'imageabilidade_1_25_ratio',
                   'imageabilidade_25_4_ratio', 'imageabilidade_4_55_ratio', 'imageabilidade_55_7_ratio',
                   'sentence_length_max', 'sentence_length_min', 'sentence_length_standard_deviation', 'verb_diversity',
                   'adj_mean', 'adj_std', 'all_mean', 'all_std', 'givenness_mean', 'givenness_std', 'span_mean',
                   'span_std', 'content_density', 'ratio_function_to_content_words']

    def rename_classe(x):
        if (x == "1_ef1"):
            return 3.0
        elif (x == '2_ef2'):
            return 8.0
        elif (x == '3_ensino_medio'):
            return 11.0
        elif (x == '4_ensino_superior'):
            return 15.0

    print(df_livros['classe'].value_counts())
    df_livros
    df_livros['classe'] = df_livros['classe'].apply(rename_classe)

    X = df_livros[feats].values
    Y = df_livros[['classe']]


    scaler = StandardScaler()
    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X)
    # scaler.mean_
    X = scaler.transform(X)


    os = RandomOverSampler(sampling_strategy='all')
    X_new, Y_new = os.fit_sample(X, Y)
    df_livros_final = pd.DataFrame(X_new, columns=feats)

    df_livros_final = pandas.DataFrame(pd.np.repeat(df_livros_final.values, 10, axis=0),
                                       columns=df_livros_final.columns)
    print(df_livros_final)


    range_index_remove = []
    cont = 0
    for i in range(1, 17):
        range_index_remove.append(cont)
        cont += 2
    print(range_index_remove)
    len(range_index_remove)
    df_livros_final = df_livros_final.drop(range_index_remove, axis=0)

    Y = Y_new

    df_livros_final['classe_norm'] = Y


    df_livros_final
    X_train, y_train = df_livros_final[feats_books], df_livros_final['classe']

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_train)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    y_train = onehot_encoded

    print(df_livros_final['classe'].value_counts())
    print(df_livros_final['classe_norm'].value_counts())
    df_livros_final.boxplot()
    print(X_train.shape)
    print(y_train.shape)
    return X_train, y_train

def load_books():
    """This function load_books is to load the extracted book metrics already stored.
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
    df_livros = pandas.read_csv("books.txt", delimiter=',', header=0)
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
             'content_density', 'ratio_function_to_content_words', 'classe']
    feats_books = ['adjective_ratio', 'adverbs', 'syllables_per_content_word', 'words_per_sentence', 'noun_ratio',
                   'pronoun_ratio', 'verbs', 'negation_ratio', 'cw_freq', 'min_cw_freq', 'first_person_pronouns', 'ttr',
                   'conn_ratio', 'add_neg_conn_ratio', 'add_pos_conn_ratio', 'cau_neg_conn_ratio', 'cau_pos_conn_ratio',
                   'log_neg_conn_ratio', 'log_pos_conn_ratio', 'tmp_neg_conn_ratio', 'tmp_pos_conn_ratio',
                   'adjectives_ambiguity', 'adverbs_ambiguity', 'nouns_ambiguity', 'verbs_ambiguity', 'yngve',
                   'frazier', 'dep_distance', 'words_before_main_verb', 'mean_noun_phrase', 'min_noun_phrase',
                   'max_noun_phrase', 'std_noun_phrase', 'passive_ratio', 'adj_arg_ovl', 'arg_ovl', 'adj_stem_ovl',
                   'stem_ovl', 'adj_cw_ovl', 'third_person_pronouns', 'concretude_mean', 'concretude_std',
                   'concretude_1_25_ratio', 'concretude_25_4_ratio', 'concretude_4_55_ratio', 'concretude_55_7_ratio',
                   'content_word_diversity', 'familiaridade_mean', 'familiaridade_std', 'familiaridade_1_25_ratio',
                   'familiaridade_25_4_ratio', 'familiaridade_4_55_ratio', 'familiaridade_55_7_ratio',
                   'idade_aquisicao_mean', 'idade_aquisicao_std', 'idade_aquisicao_1_25_ratio',
                   'idade_aquisicao_4_55_ratio', 'idade_aquisicao_55_7_ratio', 'idade_aquisicao_25_4_ratio',
                   'imageabilidade_mean', 'imageabilidade_std', 'imageabilidade_1_25_ratio',
                   'imageabilidade_25_4_ratio', 'imageabilidade_4_55_ratio', 'imageabilidade_55_7_ratio',
                   'sentence_length_max', 'sentence_length_min', 'sentence_length_standard_deviation', 'verb_diversity',
                   'adj_mean', 'adj_std', 'all_mean', 'all_std', 'givenness_mean', 'givenness_std', 'span_mean',
                   'span_std', 'content_density', 'ratio_function_to_content_words']

    print(df_livros['classe'].value_counts())
    df_livros

    X = df_livros[feats].values
    Y = df_livros[['classe']]

    ## normalizacao X
    scaler = StandardScaler()
    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X)
    # scaler.mean_
    X = scaler.transform(X)

    #

    os = RandomOverSampler(sampling_strategy='all')
    X_new, Y_new = os.fit_sample(X, Y)
    df_livros_final = pd.DataFrame(X_new, columns=feats)
    Y = Y_new
    # Y = (Y-Y.min())/(Y.max()-Y.min())
    df_livros_final['classe_norm'] = Y
    # print(df_livros_final['classe'].value_counts())
    # print(df_livros_final['classe_norm'].value_counts())

    df_livros_final
    X_train, y_train = df_livros_final[feats_books], df_livros_final['classe_norm']

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_train)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    y_train = onehot_encoded

    print(df_livros_final['classe'].value_counts())
    print(df_livros_final['classe_norm'].value_counts())
    print(X_train.shape)
    print(y_train.shape)
    return X_train, y_train

def load_books_undersample():
    """This function load_books_undersample is to load the extracted metrics already stored.
       It works as a data loading function for training, since we can change
       the data loading to other sources. Some use generic data loading functions.
       However, as we use a loading with oversampling and more transformation (Undersample)
       functions specific to books, we prefer to leave a loading itself and
       not inline or generic.

       Args:
           None

       Returns:
           Returns the metrics data (X) and their labels (Y) inside a pandas object.

       """
    df_livros = pandas.read_csv("books.txt", delimiter=',', header=0)
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
             'content_density', 'ratio_function_to_content_words', 'classe']
    feats_books = ['adjective_ratio', 'adverbs', 'syllables_per_content_word', 'words_per_sentence', 'noun_ratio',
                   'pronoun_ratio', 'verbs', 'negation_ratio', 'cw_freq', 'min_cw_freq', 'first_person_pronouns', 'ttr',
                   'conn_ratio', 'add_neg_conn_ratio', 'add_pos_conn_ratio', 'cau_neg_conn_ratio', 'cau_pos_conn_ratio',
                   'log_neg_conn_ratio', 'log_pos_conn_ratio', 'tmp_neg_conn_ratio', 'tmp_pos_conn_ratio',
                   'adjectives_ambiguity', 'adverbs_ambiguity', 'nouns_ambiguity', 'verbs_ambiguity', 'yngve',
                   'frazier', 'dep_distance', 'words_before_main_verb', 'mean_noun_phrase', 'min_noun_phrase',
                   'max_noun_phrase', 'std_noun_phrase', 'passive_ratio', 'adj_arg_ovl', 'arg_ovl', 'adj_stem_ovl',
                   'stem_ovl', 'adj_cw_ovl', 'third_person_pronouns', 'concretude_mean', 'concretude_std',
                   'concretude_1_25_ratio', 'concretude_25_4_ratio', 'concretude_4_55_ratio', 'concretude_55_7_ratio',
                   'content_word_diversity', 'familiaridade_mean', 'familiaridade_std', 'familiaridade_1_25_ratio',
                   'familiaridade_25_4_ratio', 'familiaridade_4_55_ratio', 'familiaridade_55_7_ratio',
                   'idade_aquisicao_mean', 'idade_aquisicao_std', 'idade_aquisicao_1_25_ratio',
                   'idade_aquisicao_4_55_ratio', 'idade_aquisicao_55_7_ratio', 'idade_aquisicao_25_4_ratio',
                   'imageabilidade_mean', 'imageabilidade_std', 'imageabilidade_1_25_ratio',
                   'imageabilidade_25_4_ratio', 'imageabilidade_4_55_ratio', 'imageabilidade_55_7_ratio',
                   'sentence_length_max', 'sentence_length_min', 'sentence_length_standard_deviation', 'verb_diversity',
                   'adj_mean', 'adj_std', 'all_mean', 'all_std', 'givenness_mean', 'givenness_std', 'span_mean',
                   'span_std', 'content_density', 'ratio_function_to_content_words']

    print(df_livros['classe'].value_counts())
    df_livros

    X = df_livros[feats].values
    Y = df_livros[['classe']]


    ## normalizacao X
    scaler = StandardScaler()
    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X)
    # scaler.mean_
    X = scaler.transform(X)
    #

    # os = RandomOverSampler(sampling_strategy='all')
    os = RandomUnderSampler(sampling_strategy='all')
    X_new, Y_new = os.fit_sample(X, Y)
    df_livros_final = pd.DataFrame(X_new, columns=feats)



    Y = Y_new
    # Y = (Y-Y.min())/(Y.max()-Y.min())
    df_livros_final['classe_norm'] = Y
    # print(df_livros_final['classe'].value_counts())
    # print(df_livros_final['classe_norm'].value_counts())

    df_livros_final
    X_train, y_train = df_livros_final[feats_books], df_livros_final['classe_norm']

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_train)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    y_train = onehot_encoded

    print(df_livros_final['classe'].value_counts())
    print(df_livros_final['classe_norm'].value_counts())
    print(X_train.shape)
    print(y_train.shape)

    return X_train, y_train


def load_transcricoes():
    """This function load_transcricoes is to load the extracted metrics already stored.
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
    #feats
    feats=['adjective_ratio', 'function_words', 'words_per_sentence', 'noun_ratio', 'words', 'pronoun_ratio', 'cw_freq', 'hypernyms_verbs', 'brunet', 'ttr', 'conn_ratio', 'tmp_pos_conn_ratio', 'adjectives_ambiguity', 'nouns_ambiguity', 'dep_distance', 'content_density', 'words_before_main_verb', 'anaphoric_refs', 'adj_arg_ovl', 'prepositions_per_clause', 'prepositions_per_sentence', 'infinitive_verbs', 'inflected_verbs', 'non-inflected_verbs', 'concretude_mean', 'concretude_std', 'concretude_25_4_ratio', 'concretude_4_55_ratio', 'content_word_diversity', 'content_word_standard_deviation', 'dalechall_adapted', 'familiaridade_std', 'familiaridade_25_4_ratio', 'familiaridade_4_55_ratio', 'familiaridade_55_7_ratio', 'function_word_diversity', 'idade_aquisicao_std', 'idade_aquisicao_1_25_ratio', 'idade_aquisicao_4_55_ratio', 'imageabilidade_25_4_ratio', 'noun_diversity', 'pronouns_standard_deviation', 'indicative_imperfect_ratio', 'indicative_present_ratio', 'indicative_preterite_perfect_ratio', 'oblique_pronouns_ratio', 'relative_pronouns_ratio', 'subordinate_clauses', 'coreference_pronoum_ratio', 'sentence_length_min', 'std_noun_phrase', 'verb_diversity', 'verbs_max', 'verbs_standard_deviation', 'min_freq_brwac']
    feats_tudo=['id_texto', 'Participant ID', 'School year','adjective_ratio', 'function_words', 'words_per_sentence', 'noun_ratio', 'words', 'pronoun_ratio', 'cw_freq', 'hypernyms_verbs', 'brunet', 'ttr', 'conn_ratio', 'tmp_pos_conn_ratio', 'adjectives_ambiguity', 'nouns_ambiguity', 'dep_distance', 'content_density', 'words_before_main_verb', 'anaphoric_refs', 'adj_arg_ovl', 'prepositions_per_clause', 'prepositions_per_sentence', 'infinitive_verbs', 'inflected_verbs', 'non-inflected_verbs', 'concretude_mean', 'concretude_std', 'concretude_25_4_ratio', 'concretude_4_55_ratio', 'content_word_diversity', 'content_word_standard_deviation', 'dalechall_adapted', 'familiaridade_std', 'familiaridade_25_4_ratio', 'familiaridade_4_55_ratio', 'familiaridade_55_7_ratio', 'function_word_diversity', 'idade_aquisicao_std', 'idade_aquisicao_1_25_ratio', 'idade_aquisicao_4_55_ratio', 'imageabilidade_25_4_ratio', 'noun_diversity', 'pronouns_standard_deviation', 'indicative_imperfect_ratio', 'indicative_present_ratio', 'indicative_preterite_perfect_ratio', 'oblique_pronouns_ratio', 'relative_pronouns_ratio', 'subordinate_clauses', 'coreference_pronoum_ratio', 'sentence_length_min', 'std_noun_phrase', 'verb_diversity', 'verbs_max', 'verbs_standard_deviation', 'min_freq_brwac']

    #leitura do dataframe
    df_transcricoes = pd.read_csv("metrics_file.txt", delimiter='\t', header=0)
    df_transcricoes["School year"].value_counts()
    df_n=df_transcricoes
    X = df_n[feats_tudo].values



    ## over sampling
    Y = df_n[['School year']]
    os = RandomOverSampler(sampling_strategy='all')
    X, Y = os.fit_sample(X,Y)
    ## fim do over sampling


    #criacao de dataframe para aumento de dados
    df_transcricoes_final=pd.DataFrame(X, columns=feats_tudo)
    df_transcricoes_final['School_year_norm']=Y
    print(df_transcricoes_final['School_year_norm'].value_counts())
    print(df_transcricoes_final['School year'].value_counts())

    # aumento de dados
    df_transcricoes_final_darg = pandas.DataFrame(pd.np.repeat(df_transcricoes_final.values,8,axis=0),columns=df_transcricoes_final.columns)
    print(df_transcricoes_final_darg)

    feats_55=['adjective_ratio', 'function_words', 'words_per_sentence', 'noun_ratio', 'words', 'pronoun_ratio', 'cw_freq', 'hypernyms_verbs', 'brunet', 'ttr', 'conn_ratio', 'tmp_pos_conn_ratio', 'adjectives_ambiguity', 'nouns_ambiguity', 'dep_distance', 'content_density', 'words_before_main_verb', 'anaphoric_refs', 'adj_arg_ovl', 'prepositions_per_clause', 'prepositions_per_sentence', 'infinitive_verbs', 'inflected_verbs', 'non-inflected_verbs', 'concretude_mean', 'concretude_std', 'concretude_25_4_ratio', 'concretude_4_55_ratio', 'content_word_diversity', 'content_word_standard_deviation', 'dalechall_adapted', 'familiaridade_std', 'familiaridade_25_4_ratio', 'familiaridade_4_55_ratio', 'familiaridade_55_7_ratio', 'function_word_diversity', 'idade_aquisicao_std', 'idade_aquisicao_1_25_ratio', 'idade_aquisicao_4_55_ratio', 'imageabilidade_25_4_ratio', 'noun_diversity', 'pronouns_standard_deviation', 'indicative_imperfect_ratio', 'indicative_present_ratio', 'indicative_preterite_perfect_ratio', 'oblique_pronouns_ratio', 'relative_pronouns_ratio', 'subordinate_clauses', 'coreference_pronoum_ratio', 'sentence_length_min', 'std_noun_phrase', 'verb_diversity', 'verbs_max', 'verbs_standard_deviation', 'min_freq_brwac']
    range_index_remove = []
    cont=0
    for i in range(1,141):
        range_index_remove.append(cont)
        cont+=2
    print(range_index_remove)
    len(range_index_remove)
    df_transcricoes_final_darg = df_transcricoes_final_darg.drop(range_index_remove, axis=0)

    X_train_eye, y_train_eye = df_transcricoes_final_darg[feats_55],df_transcricoes_final_darg['School_year_norm']
    ### fim do aumento de dados


    ## normalizacao X
    X = X_train_eye.values
    scaler = StandardScaler()
    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X)
    # scaler.mean_
    X_train_eye = scaler.transform(X)
    df_transcricoes_final_novo = pd.DataFrame(X_train_eye, columns=feats_55)
    X_train_eye = df_transcricoes_final_novo[feats_55]
    #

    #one hot encoding
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_train_eye)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    y_train_eye=onehot_encoded

    print(X_train_eye, y_train_eye)
    print(X_train_eye.shape, y_train_eye.shape)

    return X_train_eye, y_train_eye

def load_transcricoes_undersample():
    """This function load_transcricoes_undersample is to load the extracted metrics already stored.
         It works as a data loading function for training, since we can change
         the data loading to other sources. Some use generic data loading functions.
         However, as we use a loading with oversampling and more transformation (undersample)
         functions specific to books, we prefer to leave a loading itself and
         not inline or generic.

         Args:
             None

         Returns:
             Returns the metrics data (X) and their labels (Y) inside a pandas object.

         """
    #feats
    feats=['adjective_ratio', 'function_words', 'words_per_sentence', 'noun_ratio', 'words', 'pronoun_ratio', 'cw_freq', 'hypernyms_verbs', 'brunet', 'ttr', 'conn_ratio', 'tmp_pos_conn_ratio', 'adjectives_ambiguity', 'nouns_ambiguity', 'dep_distance', 'content_density', 'words_before_main_verb', 'anaphoric_refs', 'adj_arg_ovl', 'prepositions_per_clause', 'prepositions_per_sentence', 'infinitive_verbs', 'inflected_verbs', 'non-inflected_verbs', 'concretude_mean', 'concretude_std', 'concretude_25_4_ratio', 'concretude_4_55_ratio', 'content_word_diversity', 'content_word_standard_deviation', 'dalechall_adapted', 'familiaridade_std', 'familiaridade_25_4_ratio', 'familiaridade_4_55_ratio', 'familiaridade_55_7_ratio', 'function_word_diversity', 'idade_aquisicao_std', 'idade_aquisicao_1_25_ratio', 'idade_aquisicao_4_55_ratio', 'imageabilidade_25_4_ratio', 'noun_diversity', 'pronouns_standard_deviation', 'indicative_imperfect_ratio', 'indicative_present_ratio', 'indicative_preterite_perfect_ratio', 'oblique_pronouns_ratio', 'relative_pronouns_ratio', 'subordinate_clauses', 'coreference_pronoum_ratio', 'sentence_length_min', 'std_noun_phrase', 'verb_diversity', 'verbs_max', 'verbs_standard_deviation', 'min_freq_brwac']
    feats_tudo=['id_texto', 'Participant ID', 'School year','adjective_ratio', 'function_words', 'words_per_sentence', 'noun_ratio', 'words', 'pronoun_ratio', 'cw_freq', 'hypernyms_verbs', 'brunet', 'ttr', 'conn_ratio', 'tmp_pos_conn_ratio', 'adjectives_ambiguity', 'nouns_ambiguity', 'dep_distance', 'content_density', 'words_before_main_verb', 'anaphoric_refs', 'adj_arg_ovl', 'prepositions_per_clause', 'prepositions_per_sentence', 'infinitive_verbs', 'inflected_verbs', 'non-inflected_verbs', 'concretude_mean', 'concretude_std', 'concretude_25_4_ratio', 'concretude_4_55_ratio', 'content_word_diversity', 'content_word_standard_deviation', 'dalechall_adapted', 'familiaridade_std', 'familiaridade_25_4_ratio', 'familiaridade_4_55_ratio', 'familiaridade_55_7_ratio', 'function_word_diversity', 'idade_aquisicao_std', 'idade_aquisicao_1_25_ratio', 'idade_aquisicao_4_55_ratio', 'imageabilidade_25_4_ratio', 'noun_diversity', 'pronouns_standard_deviation', 'indicative_imperfect_ratio', 'indicative_present_ratio', 'indicative_preterite_perfect_ratio', 'oblique_pronouns_ratio', 'relative_pronouns_ratio', 'subordinate_clauses', 'coreference_pronoum_ratio', 'sentence_length_min', 'std_noun_phrase', 'verb_diversity', 'verbs_max', 'verbs_standard_deviation', 'min_freq_brwac']

    #leitura do dataframe
    df_transcricoes = pd.read_csv("metrics_file.txt", delimiter='\t', header=0)
    df_transcricoes["School year"].value_counts()
    df_n=df_transcricoes
    X = df_n[feats_tudo].values



    ## over sampling
    Y = df_n[['School year']]
    os = RandomOverSampler(sampling_strategy='all')
    #os = RandomUnderSampler(sampling_strategy='all')
    X, Y = os.fit_sample(X,Y)
    ## fim do over sampling


    #criacao de dataframe para aumento de dados
    df_transcricoes_final=pd.DataFrame(X, columns=feats_tudo)
    df_transcricoes_final['School_year_norm']=Y
    print(df_transcricoes_final['School_year_norm'].value_counts())
    print(df_transcricoes_final['School year'].value_counts())

    # aumento de dados
    df_transcricoes_final_darg = pandas.DataFrame(pd.np.repeat(df_transcricoes_final.values,3,axis=0),columns=df_transcricoes_final.columns)
    print(df_transcricoes_final_darg)

    feats_55=['adjective_ratio', 'function_words', 'words_per_sentence', 'noun_ratio', 'words', 'pronoun_ratio', 'cw_freq', 'hypernyms_verbs', 'brunet', 'ttr', 'conn_ratio', 'tmp_pos_conn_ratio', 'adjectives_ambiguity', 'nouns_ambiguity', 'dep_distance', 'content_density', 'words_before_main_verb', 'anaphoric_refs', 'adj_arg_ovl', 'prepositions_per_clause', 'prepositions_per_sentence', 'infinitive_verbs', 'inflected_verbs', 'non-inflected_verbs', 'concretude_mean', 'concretude_std', 'concretude_25_4_ratio', 'concretude_4_55_ratio', 'content_word_diversity', 'content_word_standard_deviation', 'dalechall_adapted', 'familiaridade_std', 'familiaridade_25_4_ratio', 'familiaridade_4_55_ratio', 'familiaridade_55_7_ratio', 'function_word_diversity', 'idade_aquisicao_std', 'idade_aquisicao_1_25_ratio', 'idade_aquisicao_4_55_ratio', 'imageabilidade_25_4_ratio', 'noun_diversity', 'pronouns_standard_deviation', 'indicative_imperfect_ratio', 'indicative_present_ratio', 'indicative_preterite_perfect_ratio', 'oblique_pronouns_ratio', 'relative_pronouns_ratio', 'subordinate_clauses', 'coreference_pronoum_ratio', 'sentence_length_min', 'std_noun_phrase', 'verb_diversity', 'verbs_max', 'verbs_standard_deviation', 'min_freq_brwac']
    range_index_remove = []
    cont=0
    for i in range(1,98):
        range_index_remove.append(cont)
        cont+=2
    print(range_index_remove)
    len(range_index_remove)
    df_transcricoes_final_darg = df_transcricoes_final_darg.drop(range_index_remove, axis=0)

    X_train_eye, y_train_eye = df_transcricoes_final_darg[feats_55],df_transcricoes_final_darg['School_year_norm']



    X = X_train_eye.values
    scaler = StandardScaler()
    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X)
    # scaler.mean_
    X_train_eye = scaler.transform(X)
    df_transcricoes_final_novo = pd.DataFrame(X_train_eye, columns=feats_55)
    X_train_eye = df_transcricoes_final_novo[feats_55]
    #

    #one hot encoding
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_train_eye)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    y_train_eye=onehot_encoded

    print(X_train_eye, y_train_eye)
    print(X_train_eye.shape, y_train_eye.shape)

    return X_train_eye, y_train_eye

def baseline_model_multitask():
    """This function baseline_model_multitask creates an architecture for the model based on Multi-Task-Learning Full,
     and uses the relu and softmax activation functions, and the categorical_crossentropy
     loss function.

         Args:
             None

         Returns:
             Returns the model based on a pre-built architecture for training as an object of keras 2.0

         """
    input_layer_55 = Input(shape=(55,), name='55_input')
    input_layer_79 = Input(shape=(79,), name='79_input')
    input_layer_79_d2 = Input(shape=(79,), name='79_input_d2')

    layer_1_55 = Dense(120, kernel_initializer='normal', activation='relu', name='55_layer_1')(input_layer_55)
    layer_1_79 = Dense(120, kernel_initializer='normal', activation='relu', name='79_layer_1')(input_layer_79)
    layer_1_79_d2 = Dense(120, kernel_initializer='normal', activation='relu', name='79_layer_1_d2')(input_layer_79_d2)


    merged = layers.concatenate([layer_1_55, layer_1_79, layer_1_79_d2])

    shared_layer = Dense(150, activation='relu', name='shared_layer')(merged)

    layer_3_55 = Dense(28, kernel_initializer='random_normal', activation="relu", name='55_layer_3')(shared_layer)
    layer_3_79 = Dense(28, kernel_initializer='random_normal', activation="relu", name='79_layer_3')(shared_layer)
    layer_3_79_d2 = Dense(28, kernel_initializer='random_normal', activation="relu", name='79_layer_d2_3')(shared_layer)

    layer_4_55 = Dense(14, kernel_initializer='random_normal', activation="relu", name='55_layer_4')(layer_3_55)
    layer_4_79 = Dense(14, kernel_initializer='random_normal', activation="relu", name='79_layer_4')(layer_3_79)
    layer_4_79_d2 = Dense(14, kernel_initializer='random_normal', activation="relu", name='79_layer_4_d2')(layer_3_79_d2)

    output_layer_55 = Dense(7, kernel_initializer='random_normal', activation="softmax", name='55_output')(layer_4_55)
    output_layer_79 = Dense(4, kernel_initializer='random_normal', activation="softmax", name='79_output')(layer_4_79)
    output_layer_79_d2 = Dense(4, kernel_initializer='random_normal', activation="softmax", name='79_output_d2')(layer_4_79_d2)

    model = Model(inputs=[input_layer_55, input_layer_79,input_layer_79_d2], outputs=[output_layer_55, output_layer_79,output_layer_79_d2])
    model.compile(loss='categorical_crossentropy', optimizer=tensorflow.keras.optimizers.Adam(), metrics=['accuracy'])

    return model

pipeline = baseline_model_multitask()
pipeline.summary()

X_train_eye_transcricoes, y_train_eye_transcricoes = load_transcricoes_undersample()
X_train, y_train = load_books_undersample()
X_train_d2, y_train_d2 = load_dataset1_dataset2()


total_acuracia = 0
total_fscore = 0

n_split = 10
for train_index, test_index in KFold(n_split).split(X_train):
    X_train_pss, Y_train_pss = X_train.iloc[train_index], y_train[train_index]
    X_test_pss, Y_test_pss = X_train.iloc[test_index], y_train[test_index]

    X_train_eye, Y_train_eye = X_train_eye_transcricoes.iloc[train_index], y_train_eye_transcricoes[train_index]
    X_test_eye, Y_test_eye = X_train_eye_transcricoes.iloc[test_index], y_train_eye_transcricoes[test_index]

    X_train_pss_d2, Y_train_pss_d2 = X_train_d2.iloc[train_index], y_train_d2[train_index]
    X_test_pss_d2, Y_test_pss_d2 = X_train_d2.iloc[test_index], y_train_d2[test_index]

    pipeline.fit([X_train_eye, X_train_pss, X_train_pss_d2], [Y_train_eye, Y_train_pss, Y_train_pss_d2], epochs=30, batch_size=10, verbose=0)
    prediction_tmp = pipeline.predict([X_test_eye, X_test_pss, X_test_pss_d2])
    prediction = prediction_tmp[0]

    Y_test_eye_class = []
    for i in Y_test_eye:
        Y_test_eye_class.append(np.argmax(i))

    Y_test_pss_class = []
    for i in Y_test_pss:
        Y_test_pss_class.append(np.argmax(i))

    Y_test_pss_class_d2 = []
    for i in Y_test_pss_d2:
        Y_test_pss_class_d2.append(np.argmax(i))

    prediction_class = []
    for j in range(0, len(prediction)):
        prediction_class.append(np.argmax(prediction[j]))


    print("Accuracy:", accuracy_score(Y_test_eye_class, prediction_class))
    total_acuracia += accuracy_score(Y_test_eye_class, prediction_class)
    precision, recall, fscore, support = score(Y_test_eye_class, prediction_class, average='weighted')

    print("-"*50)
    print("recall %s" % recall)
    print("precision %s" % precision)
    print("F-score %s" % fscore)
    total_fscore += fscore
    print("-"*50)


print("="*50)
print("Total Accuracy %s" % (total_acuracia / n_split))
print("Total Fscore (micro) %s" % (total_fscore / n_split))
print("="*50)
