# MTC-DTG and MTC-DTG-Simplex

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://github.com/gazzola/MTC-DTG-privado)

This repository contains the source codes and datasets used in the article *Text Complexity of Open Educational Resources in Portuguese: Mixing Written and Spoken Registers in a Multi-task Approach*. Thus, it is possible for anyone to replicate the experiments carried out in the article (submitted for evaluation in the Journal Language Resources and Evaluation [LREV] - Springer Nature).

# Corpora
+ corpora
  + books
  + transcriptions
  + MEC-RED resources

![tabelas](https://user-images.githubusercontent.com/821242/102003894-32f71700-3cea-11eb-97ac-60ea2621aefc.PNG)
![tabelas](https://github.com/gazzola/MTC-DTG-privado/blob/main/corpora/transcriptions/tabela_transcricoes.PNG?raw=true)

# Codes
Core codes were developed in Python 3.7, Tensorflow, Keras 2.0, Pandas and other libraries. They are in:

+ **Multitask Learning**
  + mtc_dtg_full.py
  + mtc_dtg_simplex.py
+ **Single Task Learning**
  + single_task_book.py
  + single_task_transcriptions.py

#### Run
For production release - single_task_book:
```sh
$ python single_task_book.py
```
For single_task_transcriptions:
```sh
$ python single_task_transcriptions.py
```
For mtc_dtg_full:
```sh
$ python mtc_dtg_full.py
```
For mtc_dtg_full:
```sh
$ python mtc_dtg_simplex.py
```
# Cite
Gazzola, M., Leal, S., Pedroni, B. et al. Text complexity of open educational resources in Portuguese: mixing written and spoken registers in a multi-task approach. Lang Resources & Evaluation 56, 621–650 (2022). https://doi.org/10.1007/s10579-021-09571-3

```bibtex
@article{Gazzola2022,
  title={Text complexity of open educational resources in Portuguese: mixing written and spoken registers in a multi-task approach},
  author={Gazzola, M. and Leal, S. and Pedroni, B. and others},
  journal={Language Resources and Evaluation},
  volume={56},
  pages={621--650},
  year={2022},
  publisher={Springer},
  doi={10.1007/s10579-021-09571-3}
}




