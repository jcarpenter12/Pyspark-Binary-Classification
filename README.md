# Pyspark Binary Classification

A small project I put together while learning Pyspark.

`pyspark_classification_functions.py` contains a list of general functions I created
whilst building a binary classification model. Used mainly for cleaning and data manipulation.

A sample project is included at the end of `pyspark_classification_functions.py`, the data for which
can be found here https://goo.gl/Zhzk1S

Clone repo with the following

```
git clone https://github.com/jcarpenter12/Pyspark-Binary-Classification.git
```

To run the binary classifier either open the `draft_notebook.ipynb` in jupyter
or run the `pyspark_classification_functions.py` as follows

With a local spark cluster running submit

```
spark-submit --master pyspark_classification_functions.py
```

or

without a cluster running

```
python pyspark_classification_functions.py
```
