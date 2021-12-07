# C4\_200M Synthetic Dataset for Grammatical Error Correction

This dataset contains synthetic training data for grammatical error correction and is described in [our BEA 2021 paper](https://www.aclweb.org/anthology/2021.bea-1.4/). To generate the parallel training data you will need to obtain the [C4 corpus](https://www.tensorflow.org/datasets/catalog/c4) first and apply the edits that are published here by following the instructions below.

## Generating the dataset

The following instructions have been tested in an
[Anaconda](https://www.anaconda.com/) (version Anaconda3 2021.05) Python
environment, but are expected to work in other Python 3 setups, too.

### Install the dependencies

Install the Abseil Python package with PIP:

```
pip install absl-py
```

### Download the C4\_200M corruptions

Change to a new working directory and download the C4\_200M corruptions from
[Kaggle Datasets](https://www.kaggle.com/felixstahlberg/the-c4-200m-dataset-for-gec):

The edits are split into 10 shards and stored as tab-separated values:

```
$ head edits.tsv-00000-of-00010

00000002020d286371dd59a2f8a900e6	8	13	is
00000002020d286371dd59a2f8a900e6	38	60	which CoinDesk says.
00000069b517cf07c79124fae6ebd0d8	0	3
00000069b517cf07c79124fae6ebd0d8	17	34	widespread dud
0000006dce3b7c10a6ad25736c173506	0	14
0000006dce3b7c10a6ad25736c173506	21	30	sales
0000006dce3b7c10a6ad25736c173506	33	44	stores
0000006dce3b7c10a6ad25736c173506	48	65	non residents are
0000006dce3b7c10a6ad25736c173506	112	120	sales tentatively
0000006dce3b7c10a6ad25736c173506	127	130	from
```

The first column is an MD5 hash that identifies a sentence in the C4 corpus. The
second and third columns are byte start and end positions, and the fourth column
contains the replacement text.


### Extract C4\_200M target sentences from C4

C4\_200M uses a relatively small subset of C4 (200M sentences). There are two ways to obtain the C4\_200M target sentences: using TensorFlow Datasets or using the C4 version provided by [allenai](https://github.com/allenai/allennlp/discussions/5056).

#### Using TensorFlow Datasets

Install the TensorFlow Datasets Python package with PIP:

```
pip install tensorflow-datasets
```

Obtain the **C4 corpus version 2.2.1** by following [these instructions](https://www.tensorflow.org/datasets/catalog/c4). The `c4200m_get_target_sentences.py` script fetches the clean target sentences from C4 for a single shard:

```
python c4200m_get_target_sentences.py edits.tsv-00000-of-00010 target_sentences.tsv-00000-of-00010 &> get_target_sentences.log-00000-of-00010
```


Repeat for the remaining nine shards, optionally with trailing ampersand for parallel processing. You can also run the concurrent script with the `concurrent-runs` parameter 
to check multiple shards at the same time.

```
python c4200m_get_target_sentences_concurrent.py edits.tsv-00000-of-00010 target_sentences.tsv-00000-of-00010 5 &> get_target_sentences.log-00000-of-00010
```

The above reads 5 shards (00000 to 00004) at once and saves the target sentences to their corresponding files.


#### Using the C4 Dataset in .json.gz Format

Given a folder containing the C4 dataset compressed in `.json.gz` files as provided by [allenai](https://github.com/allenai/allennlp/discussions/5056), it is possible to fetch the clean target sentences as follows:

```
python c4200m_get_target_sentences_json.py edits.tsv-00000-of-00010 /C4/en/target_sentences.tsv-00000-of-00010 &> get_target_sentences.log-00000-of-00010
```

where we assume the training examples of the C4 dataset are located in `/C4/en/*train*.json.gz`.

Repeat for the remaining nine shards, optionally with trailing ampersand for parallel processing.



### Apply corruption edits

The mapping from the MD5 hash to the target sentence is written to
`target_sentences.tsv*`:

```
$ head -n 3 target_sentences.tsv-00000-of-00010

00000002020d286371dd59a2f8a900e6	Bitcoin goes for $7,094 this morning, according to CoinDesk.
00000069b517cf07c79124fae6ebd0d8	1. The effect of "widespread dud" targets two face up attack position monsters on the field.
0000006dce3b7c10a6ad25736c173506	Capital Gains tax on the sale of properties for non-residents is set at 21% for 2014 and 20% in 2015 payable on profits earned on the difference of the property value between the year of purchase (purchase price plus costs) and the year of sale (sales price minus costs), based on the approved annual percentage increase on the base value approved by law.
```

To generate the final parallel dataset the edits in `edit.tsv*` have to be
applied to the sentences in `target_sentences.tsv*`:

```
python c4200m_make_sentence_pairs.py target_sentences.tsv-00000-of-00010 edits.tsv-00000-of-00010 sentence_pairs.tsv-00000-of-00010
```

The parallel data is written to `sentence_pairs.tsv*`:

```
$ head -n 3 sentence_pairs.tsv-00000-of-00010

Bitcoin is for $7,094 this morning, which CoinDesk says.	Bitcoin goes for $7,094 this morning, according to CoinDesk.
The effect of widespread dud targets two face up attack position monsters on the field.	1. The effect of "widespread dud" targets two face up attack position monsters on the field.
tax on sales of stores for non residents are set at 21% for 2014 and 20% in 2015 payable on sales tentatively earned from the difference of the property value some time of purchase (price differences according to working time) and theyear to which sale couples (sales costs), based on the approved annual on the base approved by law).	Capital Gains tax on the sale of properties for non-residents is set at 21% for 2014 and 20% in 2015 payable on profits earned on the difference of the property value between the year of purchase (purchase price plus costs) and the year of sale (sales price minus costs), based on the approved annual percentage increase on the base value approved by law.
```

Again, repeat for the remaining nine shards.

## License
The corruption edits in this dataset are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).


## BibTeX
If you found this dataset useful, please cite our [paper](https://www.aclweb.org/anthology/2021.bea-1.4/).

```
@inproceedings{stahlberg-kumar-2021-synthetic,
    title = "Synthetic Data Generation for Grammatical Error Correction with Tagged Corruption Models",
    author = "Stahlberg, Felix and Kumar, Shankar",
    booktitle = "Proceedings of the 16th Workshop on Innovative Use of NLP for Building Educational Applications",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.bea-1.4",
    pages = "37--47",
}
```

**This is not an officially supported Google product.**

