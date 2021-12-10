# -*- coding: utf-8 -*-
""" MTC-DTG Simplex
This file represents the implementation of the Multi-Task Learning (MTL) implementations with two tasks,
called Multi-task approach to Text Complexity using Different Text Genres (MTC-DTG) Simplex,
and with three tasks, called Multi-task approach to Text Complexity using Different Text Genres (MTC-DTG)
In this work, we assessed the effect of class balancing with an oversampling and undersampling approach.
We have also used data augmentation techniques, which were necessary for MTL architectures to have equal data entry.

The files used, such as books.txt, metrics.txt, among others. These are metrics extracted from the corpora using NILC Metrics.

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
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from keras import Model, layers
from sklearn.metrics import accuracy_score
from keras.layers import Dense, Input
import numpy as np
from keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support as score

df_transcricoes = pd.read_csv("metrics_file.txt", delimiter='\t', header=0)
df_n=df_transcricoes
feats_tudo=['id_texto', 'Participant ID', 'School year','adjective_ratio', 'function_words', 'words_per_sentence', 'noun_ratio', 'words', 'pronoun_ratio', 'cw_freq', 'hypernyms_verbs', 'brunet', 'ttr', 'conn_ratio', 'tmp_pos_conn_ratio', 'adjectives_ambiguity', 'nouns_ambiguity', 'dep_distance', 'content_density', 'words_before_main_verb', 'anaphoric_refs', 'adj_arg_ovl', 'prepositions_per_clause', 'prepositions_per_sentence', 'infinitive_verbs', 'inflected_verbs', 'non-inflected_verbs', 'concretude_mean', 'concretude_std', 'concretude_25_4_ratio', 'concretude_4_55_ratio', 'content_word_diversity', 'content_word_standard_deviation', 'dalechall_adapted', 'familiaridade_std', 'familiaridade_25_4_ratio', 'familiaridade_4_55_ratio', 'familiaridade_55_7_ratio', 'function_word_diversity', 'idade_aquisicao_std', 'idade_aquisicao_1_25_ratio', 'idade_aquisicao_4_55_ratio', 'imageabilidade_25_4_ratio', 'noun_diversity', 'pronouns_standard_deviation', 'indicative_imperfect_ratio', 'indicative_present_ratio', 'indicative_preterite_perfect_ratio', 'oblique_pronouns_ratio', 'relative_pronouns_ratio', 'subordinate_clauses', 'coreference_pronoum_ratio', 'sentence_length_min', 'std_noun_phrase', 'verb_diversity', 'verbs_max', 'verbs_standard_deviation', 'min_freq_brwac']
feats=['adjective_ratio', 'function_words', 'words_per_sentence', 'noun_ratio', 'words', 'pronoun_ratio', 'cw_freq', 'hypernyms_verbs', 'brunet', 'ttr', 'conn_ratio', 'tmp_pos_conn_ratio', 'adjectives_ambiguity', 'nouns_ambiguity', 'dep_distance', 'content_density', 'words_before_main_verb', 'anaphoric_refs', 'adj_arg_ovl', 'prepositions_per_clause', 'prepositions_per_sentence', 'infinitive_verbs', 'inflected_verbs', 'non-inflected_verbs', 'concretude_mean', 'concretude_std', 'concretude_25_4_ratio', 'concretude_4_55_ratio', 'content_word_diversity', 'content_word_standard_deviation', 'dalechall_adapted', 'familiaridade_std', 'familiaridade_25_4_ratio', 'familiaridade_4_55_ratio', 'familiaridade_55_7_ratio', 'function_word_diversity', 'idade_aquisicao_std', 'idade_aquisicao_1_25_ratio', 'idade_aquisicao_4_55_ratio', 'imageabilidade_25_4_ratio', 'noun_diversity', 'pronouns_standard_deviation', 'indicative_imperfect_ratio', 'indicative_present_ratio', 'indicative_preterite_perfect_ratio', 'oblique_pronouns_ratio', 'relative_pronouns_ratio', 'subordinate_clauses', 'coreference_pronoum_ratio', 'sentence_length_min', 'std_noun_phrase', 'verb_diversity', 'verbs_max', 'verbs_standard_deviation', 'min_freq_brwac']
X = df_n[feats_tudo].values
df_transcricoes["School year"].value_counts()
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
X = df_n[feats_tudo].values
Y = df_n[['School year']]
os = RandomOverSampler(sampling_strategy='all')
X, Y = os.fit_sample(X,Y)

df_transcricoes_final=pd.DataFrame(X, columns=feats_tudo)
df_transcricoes_final['School_year_norm']=Y
df_transcricoes_final['School_year_norm'].value_counts()
df_transcricoes_final['School year'].value_counts()
df_transcricoes_final_darg = pandas.DataFrame(pd.np.repeat(df_transcricoes_final.values,8,axis=0),columns=df_transcricoes_final.columns)
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

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_train_eye)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
y_train_eye=onehot_encoded

print(X_train_eye, y_train_eye)
X_train_eye_transcricoes, y_train_eye_transcricoes = X_train_eye, y_train_eye
print(X_train_eye.shape, y_train_eye.shape)

print(X_train_eye_transcricoes.shape, y_train_eye_transcricoes.shape)

df_livros = pandas.read_csv("books.txt", delimiter=',', header=0)
feats=['adjective_ratio','adverbs','syllables_per_content_word','words_per_sentence','noun_ratio','pronoun_ratio','verbs','negation_ratio','cw_freq','min_cw_freq','first_person_pronouns','ttr','conn_ratio','add_neg_conn_ratio','add_pos_conn_ratio','cau_neg_conn_ratio','cau_pos_conn_ratio','log_neg_conn_ratio','log_pos_conn_ratio','tmp_neg_conn_ratio','tmp_pos_conn_ratio','adjectives_ambiguity','adverbs_ambiguity','nouns_ambiguity','verbs_ambiguity','yngve','frazier','dep_distance','words_before_main_verb','mean_noun_phrase','min_noun_phrase','max_noun_phrase','std_noun_phrase','passive_ratio','adj_arg_ovl','arg_ovl','adj_stem_ovl','stem_ovl','adj_cw_ovl','third_person_pronouns','concretude_mean','concretude_std','concretude_1_25_ratio','concretude_25_4_ratio','concretude_4_55_ratio','concretude_55_7_ratio','content_word_diversity','familiaridade_mean','familiaridade_std','familiaridade_1_25_ratio','familiaridade_25_4_ratio','familiaridade_4_55_ratio','familiaridade_55_7_ratio','idade_aquisicao_mean','idade_aquisicao_std','idade_aquisicao_1_25_ratio','idade_aquisicao_4_55_ratio','idade_aquisicao_55_7_ratio','idade_aquisicao_25_4_ratio','imageabilidade_mean','imageabilidade_std','imageabilidade_1_25_ratio','imageabilidade_25_4_ratio','imageabilidade_4_55_ratio','imageabilidade_55_7_ratio','sentence_length_max','sentence_length_min','sentence_length_standard_deviation','verb_diversity','adj_mean','adj_std','all_mean','all_std','givenness_mean','givenness_std','span_mean','span_std','content_density','ratio_function_to_content_words','classe']
feats_books=['adjective_ratio','adverbs','syllables_per_content_word','words_per_sentence','noun_ratio','pronoun_ratio','verbs','negation_ratio','cw_freq','min_cw_freq','first_person_pronouns','ttr','conn_ratio','add_neg_conn_ratio','add_pos_conn_ratio','cau_neg_conn_ratio','cau_pos_conn_ratio','log_neg_conn_ratio','log_pos_conn_ratio','tmp_neg_conn_ratio','tmp_pos_conn_ratio','adjectives_ambiguity','adverbs_ambiguity','nouns_ambiguity','verbs_ambiguity','yngve','frazier','dep_distance','words_before_main_verb','mean_noun_phrase','min_noun_phrase','max_noun_phrase','std_noun_phrase','passive_ratio','adj_arg_ovl','arg_ovl','adj_stem_ovl','stem_ovl','adj_cw_ovl','third_person_pronouns','concretude_mean','concretude_std','concretude_1_25_ratio','concretude_25_4_ratio','concretude_4_55_ratio','concretude_55_7_ratio','content_word_diversity','familiaridade_mean','familiaridade_std','familiaridade_1_25_ratio','familiaridade_25_4_ratio','familiaridade_4_55_ratio','familiaridade_55_7_ratio','idade_aquisicao_mean','idade_aquisicao_std','idade_aquisicao_1_25_ratio','idade_aquisicao_4_55_ratio','idade_aquisicao_55_7_ratio','idade_aquisicao_25_4_ratio','imageabilidade_mean','imageabilidade_std','imageabilidade_1_25_ratio','imageabilidade_25_4_ratio','imageabilidade_4_55_ratio','imageabilidade_55_7_ratio','sentence_length_max','sentence_length_min','sentence_length_standard_deviation','verb_diversity','adj_mean','adj_std','all_mean','all_std','givenness_mean','givenness_std','span_mean','span_std','content_density','ratio_function_to_content_words']

print(df_livros['classe'].value_counts())

X = df_livros[feats].values
Y = df_livros[['classe']]

scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
os = RandomOverSampler(sampling_strategy='all')
X_new, Y_new = os.fit_sample(X,Y)
df_livros_final=pd.DataFrame(X_new, columns=feats)
Y = Y_new
df_livros_final['classe_norm']=Y
X_train, y_train = df_livros_final[feats_books], df_livros_final['classe_norm']

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_train)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

print(df_livros_final['classe'].value_counts())
print(df_livros_final['classe_norm'].value_counts())
print(X_train.shape)
print(y_train.shape)


def baseline_model_multitask():
    """This function baseline_model_multitask creates an architecture for the model based on Multi-Task-Learning Simplex,
     and uses the relu and softmax activation functions, and the categorical_crossentropy
     loss function.

         Args:
             None

         Returns:
             Returns the model based on a pre-built architecture for training as an object of keras 2.0

         """
    input_layer_55 = Input(shape=(55,), name='55_input')
    input_layer_79 = Input(shape=(79,), name='79_input')

    layer_1_55 = Dense(120, kernel_initializer='normal', activation='relu', name='55_layer_1')(input_layer_55)
    layer_1_79 = Dense(120, kernel_initializer='normal', activation='relu', name='79_layer_1')(input_layer_79)

    merged = layers.concatenate([layer_1_55, layer_1_79])

    shared_layer = Dense(120, activation='relu', name='shared_layer')(merged)

    layer_3_55 = Dense(28, kernel_initializer='random_normal', activation="relu", name='55_layer_3')(shared_layer)
    layer_3_79 = Dense(28, kernel_initializer='random_normal', activation="relu", name='79_layer_3')(shared_layer)

    layer_4_55 = Dense(14, kernel_initializer='random_normal', activation="relu", name='55_layer_4')(layer_3_55)
    layer_4_79 = Dense(14, kernel_initializer='random_normal', activation="relu", name='79_layer_4')(layer_3_79)

    output_layer_55 = Dense(7, kernel_initializer='random_normal', activation="softmax", name='55_output')(layer_4_55)
    output_layer_79 = Dense(4, kernel_initializer='random_normal', activation="softmax", name='79_output')(layer_4_79)

    model = Model(inputs=[input_layer_55, input_layer_79], outputs=[output_layer_55, output_layer_79])
    model.compile(loss='categorical_crossentropy', optimizer=tensorflow.keras.optimizers.Adam(), metrics=['accuracy'])

    return model





pipeline = baseline_model_multitask()
pipeline.summary()

total_acuracia = 0
total_fscore = 0
n_split = 10
for train_index, test_index in KFold(n_split).split(X_train):
    X_train_pss, Y_train_pss = X_train.iloc[train_index], y_train[train_index]
    X_test_pss, Y_test_pss = X_train.iloc[test_index], y_train[test_index]
    X_train_eye, Y_train_eye = X_train_eye_transcricoes.iloc[train_index], y_train_eye[train_index]
    X_test_eye, Y_test_eye = X_train_eye_transcricoes.iloc[test_index], y_train_eye[test_index]

    pipeline.fit([X_train_eye, X_train_pss], [Y_train_eye, Y_train_pss], epochs=30, batch_size=10, verbose=0)
    prediction_tmp = pipeline.predict([X_test_eye, X_test_pss])
    prediction = prediction_tmp[1]

    Y_test_eye_class = []
    for i in Y_test_eye:
        Y_test_eye_class.append(np.argmax(i))

    Y_test_pss_class = []
    for i in Y_test_pss:
        Y_test_pss_class.append(np.argmax(i))

    prediction_class = []
    for j in range(0, len(prediction)):
        prediction_class.append(np.argmax(prediction[j]))

    print("Accuracy:", accuracy_score(Y_test_pss_class, prediction_class))

    total_acuracia += accuracy_score(Y_test_pss_class, prediction_class)
    precision, recall, fscore, support = score(Y_test_pss_class, prediction_class, average='micro')

    print("---------------")
    print("Recall %s" % recall)
    print("Precision %s" % precision)
    # print("specificity %s" % specificity)
    print("F-score %s" % fscore)
    total_fscore += fscore
    print("---------------")

print("=================")
print("Total Accuracy %s" % (total_acuracia / n_split))
print("Total F-score (micro) %s" % (total_fscore / n_split))
print("=================")

