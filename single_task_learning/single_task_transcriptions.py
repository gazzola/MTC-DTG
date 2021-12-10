# -*- coding: utf-8 -*-
""" Single Task Learning (STL) for transcriptions
This file represents the implementation of the Single Task Learning used in the research.
Single Task Learning (STL) approach and two Multi-Task Learning (MTL) implementations with two tasks,
called Multi-task approach to Text Complexity using Different Text Genres (MTC-DTG) Simplex,
and with three tasks, called Multi-task approach to Text Complexity using Different Text Genres (MTC-DTG)
In this work, we assessed the effect of class balancing with an oversampling and undersampling approach.
We have also used data augmentation techniques, which were necessary for MTL architectures to have equal data entry.

The files used, such as books_only_metrics_run.txt, dataset1_dataset2.csv, books.txt, metrics.txt, among others. These are metrics extracted from the corpora using NILC Metrics.

Example:
        To run this experiment, you must extract the metrics, if you haven't already,
        and insert in a CSV and use the names of the features presented in
        def load_livros_single. The list of features needed as feats.

        $ python single_task_transcriptions.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Todo:
    * For module TODOs
    *

.. _NILC-ICMC USP Group - Other tools:
   http://www.nilc.icmc.usp.br/nilc/index.php/tools-and-resources

"""
import numpy
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from numpy import argmax
from keras.utils import to_categorical
from scipy.stats import zscore
from imblearn.over_sampling import RandomOverSampler
import collections


pd.set_option('display.max_colwidth', -1)

# Load dataset
df = pd.read_csv("metrics.txt", delimiter='\t', header=0)

# 55 metrics
feats=['adjective_ratio', 'function_words', 'words_per_sentence', 'noun_ratio', 'words', 'pronoun_ratio', 'cw_freq', 'hypernyms_verbs', 'brunet', 'ttr', 'conn_ratio', 'tmp_pos_conn_ratio', 'adjectives_ambiguity', 'nouns_ambiguity', 'dep_distance', 'content_density', 'words_before_main_verb', 'anaphoric_refs', 'adj_arg_ovl', 'prepositions_per_clause', 'prepositions_per_sentence', 'infinitive_verbs', 'inflected_verbs', 'non-inflected_verbs', 'concretude_mean', 'concretude_std', 'concretude_25_4_ratio', 'concretude_4_55_ratio', 'content_word_diversity', 'content_word_standard_deviation', 'dalechall_adapted', 'familiaridade_std', 'familiaridade_25_4_ratio', 'familiaridade_4_55_ratio', 'familiaridade_55_7_ratio', 'function_word_diversity', 'idade_aquisicao_std', 'idade_aquisicao_1_25_ratio', 'idade_aquisicao_4_55_ratio', 'imageabilidade_25_4_ratio', 'noun_diversity', 'pronouns_standard_deviation', 'indicative_imperfect_ratio', 'indicative_present_ratio', 'indicative_preterite_perfect_ratio', 'oblique_pronouns_ratio', 'relative_pronouns_ratio', 'subordinate_clauses', 'coreference_pronoum_ratio', 'sentence_length_min', 'std_noun_phrase', 'verb_diversity', 'verbs_max', 'verbs_standard_deviation', 'min_freq_brwac']

X = df[feats].values
Y = df[['School year']]


# One hot encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(Y.values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
Y=onehot_encoded

scaler = StandardScaler()
#scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X)
#scaler.mean_
X=scaler.transform(X)

os = RandomOverSampler(sampling_strategy='all')
X_new, Y_new = os.fit_sample(X,Y)

df_transcricoes_final=pd.DataFrame(X_new, columns=feats)
X_new_df=df_transcricoes_final[feats]

### Returns list of class  ###
df_2 = pd.read_csv("metrict3_.txt", delimiter='\t', header=0)

# 55 feats
feats=['adjective_ratio', 'function_words', 'words_per_sentence', 'noun_ratio', 'words', 'pronoun_ratio', 'cw_freq', 'hypernyms_verbs', 'brunet', 'ttr', 'conn_ratio', 'tmp_pos_conn_ratio', 'adjectives_ambiguity', 'nouns_ambiguity', 'dep_distance', 'content_density', 'words_before_main_verb', 'anaphoric_refs', 'adj_arg_ovl', 'prepositions_per_clause', 'prepositions_per_sentence', 'infinitive_verbs', 'inflected_verbs', 'non-inflected_verbs', 'concretude_mean', 'concretude_std', 'concretude_25_4_ratio', 'concretude_4_55_ratio', 'content_word_diversity', 'content_word_standard_deviation', 'dalechall_adapted', 'familiaridade_std', 'familiaridade_25_4_ratio', 'familiaridade_4_55_ratio', 'familiaridade_55_7_ratio', 'function_word_diversity', 'idade_aquisicao_std', 'idade_aquisicao_1_25_ratio', 'idade_aquisicao_4_55_ratio', 'imageabilidade_25_4_ratio', 'noun_diversity', 'pronouns_standard_deviation', 'indicative_imperfect_ratio', 'indicative_present_ratio', 'indicative_preterite_perfect_ratio', 'oblique_pronouns_ratio', 'relative_pronouns_ratio', 'subordinate_clauses', 'coreference_pronoum_ratio', 'sentence_length_min', 'std_noun_phrase', 'verb_diversity', 'verbs_max', 'verbs_standard_deviation', 'min_freq_brwac']
print(len(feats))

Y_2 = df_2[['School year']]

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(Y_2.values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
Y_2=onehot_encoded

decode_label=[]
for i in range(len(Y_new)):
    decode_label.append(label_encoder.inverse_transform([argmax(Y_new[i, :])]))



unique, counts = numpy.unique([decode_label], return_counts=True)
lista_classes_count=dict(zip(unique, counts))


def baseline_model_55feat():
    """This function creates an architecture for the model based on Single Task Learning,
    and uses the relu and softmax activation functions, and the categorical_crossentropy
    loss function.

        Args:
            None

        Returns:
            Returns the model based on a pre-built architecture for training as an object of keras 2.0

        """
    model = Sequential()
    model.add(Dense(120, input_dim=55, activation='relu',
                    kernel_initializer='random_normal'))
    model.add(Dense(28, activation='relu', kernel_initializer='random_normal'))
    model.add(Dense(14, activation='relu', kernel_initializer='random_normal'))
    model.add(Dense(7, activation='softmax',
                    kernel_initializer='random_normal'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model_55feat, epochs=100, batch_size=10, verbose=1)
kfold = KFold(n_splits=10, shuffle=True, random_state=2020)
results = cross_val_score(estimator, X_new, Y_new, cv=kfold, verbose=1)
print("Result: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

pipeline = baseline_model_55feat()
total_acuracia = 0
total_fscore = 0
n_split = 10
for train_index, test_index in KFold(n_split).split(X_new_df):
    X_train_pss, Y_train_pss = X_new_df.iloc[train_index], Y_new[train_index]
    X_test_pss, Y_test_pss = X_new_df.iloc[test_index], Y_new[test_index]

    pipeline.fit(X_train_pss, Y_train_pss, epochs=30, batch_size=10, verbose=0)
    prediction = pipeline.predict(X_test_pss)
    Y_test_pss_class = []
    for i in Y_test_pss:
        Y_test_pss_class.append(np.argmax(i))

    prediction_class = []
    for j in range(0, len(prediction)):
        prediction_class.append(np.argmax(prediction[j]))
    print("Accuracy:", accuracy_score(Y_test_pss_class, prediction_class))
    total_acuracia += accuracy_score(Y_test_pss_class, prediction_class)
    precision, recall, fscore, support = score(Y_test_pss_class, prediction_class, average='micro')

print("=================")
print("Total Accuracy %s" % (total_acuracia / n_split))
print("Total F-score (micro) %s" % (total_fscore / n_split))
print("=================")
