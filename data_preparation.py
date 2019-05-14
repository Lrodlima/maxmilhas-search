# -*- coding: utf-8 -*-
from pyspark import SparkContext
from pyspark.sql.functions import when, col
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
#from pyspark.ml.evaluation as evals
from pyspark.sql.types import *
from pyspark.sql import SparkSession
import argparse

# Create spark Session
spark = SparkSession \
            .builder \
            .appName('MaxMilhas <> Processo Seletivo') \
            .getOrCreate()


parser = argparse.ArgumentParser(description="")
parser.add_argument('-dp', '--datapath', action="store", help="Store the input data path")
parser.add_argument('-op', '--outpath', action="store", help="Store the output data path")

args = parser.parse_args()

if args.datapath:
    input_path = args.datapath

    fields = [
        StructField("id", StringType(), True),
        StructField("idusers", FloatType(), True),
        StructField("idsearch", StringType(), True),
        StructField("airline", StringType(), True),
        StructField("tipo_de_voo", StringType(), True),
        StructField("origin", StringType(), True),
        StructField("dest", StringType(), True),
        StructField("dep_date", DateType(), True),
        StructField("arr_date", DateType(), True),
        StructField("days_travelled", IntegerType(), False),
        StructField("adults", IntegerType(), True),
        StructField("childs", IntegerType(), True),
        StructField("infants", IntegerType(), True),
        StructField("dep_country", StringType(), True),
        StructField("arr_country", StringType(), True),
        StructField("international_flight", StringType(), False),
        StructField("direction", StringType(), True),
        StructField("class", StringType(), True),
        StructField("search_date", DateType(), True),
        StructField("wanted_dep_date", DateType(), True),
        StructField("wanted_arr_date", DateType(), True),
        StructField("search_received_date", DateType(), True),
        StructField("seconds_search", IntegerType(), True),
        StructField("seconds_to_results", IntegerType(), True),
        StructField("quant_flights", IntegerType(), True),
        StructField("quant_flights_received", IntegerType(), True),
        StructField("quant_best_price_airlines", FloatType(), True),
        StructField("quant_best_price_mm", FloatType(), True),
        StructField("cheapest_mm_price_dep", StringType(), True),
        StructField("cheapest_mm_price_arr", StringType(), False),
        StructField("airport_dep_name", StringType(), True),
        StructField("airport_code_dep", StringType(), True),
        StructField("airport_dep_combination", StringType(), True),
        StructField("airport_dep_group", StringType(), True),
        StructField("airport_arr_name", StringType(), True),
        StructField("airport_code_arr", StringType(), True),
        StructField("airport_arr_combination", StringType(), True),
        StructField("airport_arr_group", StringType(), True)
    ]

    schema = StructType(fields)

    max_df = spark.read \
        .option("delimiter", ";") \
        .option("encoding", "ISO-8859-1") \
        .csv(input_path+"in.csv",
             header=False,
             schema=schema)

    max_df = max_df.withColumn('is_logged', when(col('idusers').isNull(), False).otherwise(True))
    max_df = max_df.withColumn("is_logged", max_df.is_logged.cast("integer"))
    max_df = max_df.withColumn('label', when(col('international_flight') == "NÃO", 0).otherwise(1).cast("integer"))

    # Columns that can be dropped
    cols_to_drop = [
        'id',
        'idusers',
        'idsearch',
        'airport_dep_name',
        'airport_arr_name',
        'airport_code_arr',
        'airport_code_dep',
        'airport_dep_group',
        'airport_arr_group',
        'airport_dep_combination',
        'airport_arr_combination',
        'direction',
        'cheapest_mm_price_dep',
        'cheapest_mm_price_arr',
        'international_flight' # Label variable
    ]

    max_df = max_df.drop(*cols_to_drop)

    max_df.write.parquet(args.outpath, mode="overwrite")

    # Columns with categorical values
    categorical_cols = [
        'airline',
        'class',
        'tipo_de_voo',
        'origin',
        'dest',
        'dep_country',
        'arr_country'
    ]

    stages = [] # stages in our Pipeline
    for cat_col in categorical_cols:
        # Category Indexing with StringIndexer
        string_indexer = StringIndexer(inputCol=cat_col, outputCol=cat_col + "_index")

        # Use OneHotEncoder to convert categorical variables into binary SparseVectors
        encoder = OneHotEncoderEstimator(inputCols=[string_indexer.getOutputCol()], outputCols=[cat_col + "_fact"])

        # Add stages.These are not run here, but will run all at once later on.
        stages += [string_indexer, encoder]

    # Transform all features into a vector using VectorAssembler
    numericCols = ["days_travelled", "adults", "childs", "infants", "seconds_search", 
                   "seconds_to_results", "quant_flights", "quant_flights_received", 
                   "quant_best_price_airlines", "quant_best_price_mm", "is_logged"]
    assemblerInputs = [c + "_fact" for c in categorical_cols] + numericCols
    vec_assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [vec_assembler]

    # Make the pipeline
    mm_pipe = Pipeline().setStages(stages)

    # Fit and transform the data
    pipelineModel = mm_pipe.fit(max_df)
    preDataDF = pipelineModel.transform(max_df)

    # Split the data into training and test sets
    training, test = preDataDF.randomSplit([.7, .3])

    lr = LogisticRegression()

    #evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")

from IPython import embed; embed()
