from pyspark import SparkContext, SparkConf
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import UserDefinedFunction
from pyspark.mllib.linalg import Vectors
import os
import pyspark_csv as pycsv

os.environ["SPARK_LOCAL_IP"] = "127.0.1.1" # set local IP
print('GOT HERE')

def create_spark_context():


    conf = (SparkConf()
            .setMaster('local')
            .setAppName('RfClassifier')
            .set("spark.executor.memory","2g"))
    sc = SparkContext(conf=conf)

    return sc

def create_dataframes(directory,schema_train=None,schema_test=None):
    """
    Creates dataframes from directory
    Must be named 'train' or 'test'.
    Returns only train if test N/A

    Inputs: directory - String, schema defaults to false
    and will infer from input .csv else will apply
    specified schema/schemas

    Returns: Dataframes/Dataframe

    """
    inferSchema = True if schema_train==None else False
    if schema_test==None:
        schema_test = schema_train

    if os.path.exists(directory):
        train_path = directory+"/train.csv"
        if os.path.exists(train_path):
            df_train = sql.read.csv(train_path,
                         header = True,
                         inferSchema = inferSchema,
                         schema=schema_train)
        else:
            raise ValueError("train.csv not found in %s" % directory)

        test_path = directory+"/test.csv"
        if os.path.exists(test_path):
            df_test = sql.read.csv(test_path,
                         header = True,
                         inferSchema = inferSchema,
                         schema=schema_test)

            return df_train,df_test

        return df_train

    else:
        raise ValueError("%s does not exist" % directory)


def combine_train_test(df_train,df_test,label):
    """
    Combine train and test dataframes
    Creates dummy column if label not in test

    inputs: df_train - Spark DataFrame, df_test - Spark DataFrame,
    label - String

    returns: DataFrame

    """
    #Mark dataframes
    df_train = df_train.withColumn('Mark',lit('train'))
    df_test  = df_test.withColumn('Mark',lit('test'))

    def has_column(df, column):
        try:
            df[column]
            return True
        except AnalysisException:
            return False

    if has_column(df_test,label):
        if len(df_train.columns) == len(df_test.columns):
            #rearrange columns to avoid mis label when grouping together
            df_test = df_test.select(df_train.columns)
            return (df_train.union(df_test))
        else:
            raise ValueError("input dataframes of different shape")
    else:
        #add dummy label column to dataframe
        df_test = df_test.withColumn(label,lit(0))
        if len(df_train.columns) == len(df_test.columns):
            df_test = df_test.select(df_train.columns)
            return (df_train.union(df_test))
        else:
            raise ValueError("input dataframes of different shape")

def get_missing(df):
    """
    Prints no. missing values for each column

    inputs: df - Spark DataFrame

    returns: None

    """

    for column in df.columns:
        missing = df.where(df[column].isNull()).count()
        print("Missing values for %s : %s" % (column,missing))

    return None

def remove_missing_columns(df,thresh=0.05,ignore=[]):
    """
    Removes column from dataframe if the column
    has higher number of null values than thresh


    inputs: DataFrame, float - thresh (defaults to 0.05),
    ignore - Array (list of columns to be exempt)

    returns: DataFrame
    """

    x = df.cache()

    columns = filter(lambda x: x not in ignore,x.columns)

    for column in columns:
        missing = df.where(df[column].isNull()).count()
        if missing != 0:
            if (missing/x.count()) > thresh:
                x=x.drop(column)

    return x

def fill_null_with_mean(df):
    """
    Replaces null numeric values with
    mean value
    Replaces null categorical string values
    with mode
    input: spark dataframe
    returns: spark dataframe

    """

    x = df.cache()

    for column in df.schema.fields:
        if df.where(df[column.name].isNull()).count() > 0:

            dtype = "%s" % column.dataType
            if dtype != "StringType":
                mean = df.groupBy().mean(column.name).first()[0]
                x = x.na.fill({column.name:mean})
            else:
                counts = df.groupBy(column.name).count()
                mode = counts.join(
                counts.agg(F.max("count").alias("max_")),
                col("count") == col("max_")
                ).limit(1).select(column.name)
                x = x.na.fill({column.name:mode.first()[0]})
    return x

def build_pipeline(df,label):
    """
    Build pipeline to fit and transform data on

    Inputs: df - spark DataFrame, label - String relating label column

    Returns: pipeline and CrossValidator object

    """

    categorical = []
    numeric = []


    for column in df.schema.fields:
        if column.name != label and column.name !='Mark':
            cType = "%s" % column.dataType
            if cType == "StringType":
                categorical.append(column.name)
            else:
                numeric.append(column.name)

    indexers = [StringIndexer(inputCol=column,
                              outputCol=column+"_index")\
                              for column in categorical]
    labelIndexer = StringIndexer(inputCol=label,outputCol=label+"_index").fit(df)
    index_categorical = [column + "_index" for column in categorical]
    all_columns = index_categorical + numeric

    assembler = VectorAssembler(inputCols=all_columns,outputCol="features")
    scaler = MinMaxScaler(inputCol="features",outputCol="scaledFeatures")

    rf = RandomForestClassifier(labelCol=label+"_index",
                            featuresCol="scaledFeatures",
                            numTrees=10,
                            maxBins=200)

    #Used to convert predicted values back to their original format
    labelConverter = IndexToString(inputCol="prediction",
                                   outputCol="predictedLabel",
                                   labels=labelIndexer.labels)


    #assembler is added to list with square brackets
    stages = indexers + [labelIndexer,assembler,scaler,rf,labelConverter]
    pipeline = Pipeline(stages = stages)

    paramGrid = ParamGridBuilder()\
                .addGrid(rf.maxBins,[25,50,75])\
                .addGrid(rf.maxDepth,[4,6,8])\
                .build()

    crossVal = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=BinaryClassificationEvaluator(),
                              numFolds=4)

    return pipeline,crossVal

def split_into_train_test(df,train_sample_size=0.7):
    """
    Splits a dataframe into train and test.
    If dataframe contains no column 'Mark'
    it splits on default 0.7/0.3 random sampling

    inputs: df - Spark DataFrame, train_sample_size - float (0-1)

    returns: train - Spark DataFrame, test - Spark DataFrame
    """

    def has_column(df, col):
        try:
            df[col]
            return True
        except AnalysisException:
            return False

    if has_column(df,'Mark'):
        train = df.where(df['Mark']=='train')
        train = train.drop('Mark')
        test  = df.where(df['Mark']=='test')
        test = test.drop('Mark')

    else:
        if train_sample_size > 1 or train_sample_size < 0:
            raise ValueError("train_sample_size out of bounds")
        test_sample_size = 1 - train_sample_size
        (train,test) = df.randomSplit([train_sample_size,
                                       test_sample_size])

    return train,test

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    sc = SparkContext(appName="Test")
    print("THIS IS A TEST")
    print(sc)
    sc.stop()
