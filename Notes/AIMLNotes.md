### AI ML notes
| Ref: Reference: https://ml-course.github.io/master/notebooks/01%20-%20Introduction.html

# Data
### Data collection

Traits of good data:
1. Large datasets.
2. Precise attribute types, feature rich.
3. Complete fields, no missing values.
4. Values are consistent.
5. Solid distribution of outcomes.
6. Fair sampling

Traits of bad data:
1. Small datasets (less than 100 rows).
2. Useless attributes, not needed for solving problem at hand.
3. Missing values, null fields.
4. Inconsistent values.
5. Lots of positive outcome, few negative outcomes.
6. Biased sampling

Why:
1. Generally, more data means better model training.
2. Models need to train on important features.
3. Models can skew results when data points are missing.
4. Models like clean and consistent data.
5. Models cannot learn with skewed distributions of outcomes.
6. Models will skew results with biased data.

#### Traits of Good Data vs Bad Data

| Traits of good data | Traits of bad data | Why |
|-|-|-|
| Large datasets. | Small datasets (less than 100 rows). | Generally, more data means better model training. |
| Precise attribute types, feature rich. | Useless attributes, not needed for solving problem at hand. | Models need to train on important features. |
| Complete fields, no missing values. | Missing values, null fields. | Models can skew results when data points are missing. |
| Values are consistent. | Inconsistent values. | Models like clean and consistent data. |
| Solid distribution of outcomes. | Lots of positive outcome, few negative outcomes. | Models cannot learn with skewed distributions of outcomes. |
| Fair sampling | Biased sampling | Models will skew results with biased data. |

Collecting, Cleaning and Preparing data is the most time consuming aspect on ML.
- Where does the data come from? 

### Data terminologies
- **Dataset** is a collection of data and it is same as the inputdata or training/testing data. Dataset can be column, images, json or video
- **Features** is as same as the column/attributes in a dataset
- **Row/Observations** is the different datapoints that make up the datasets, for example the row called "Ramu" in the column Name - in which Ramu is the row/Observation. IN other words one row in the structured data.

### Types of Data
- Structured Data 
      - With clear tables, row and columns.
      - Data has a defined schema. A schema is the information about the data, like attributes name and data type that describes the data, like string or int etc. 
- Semi Structured Data
    - CSV, JSON
- Unstructured.
      - App logs, Text music and video
      - No defined schema
### Types of data used in ML 
- **Labeled Data**
    - The data where we already know what is the target attribute. Example assume you have to categorize SPAM/not SPAM emails, you will create a data set with sender domain, subject and email contents. This data set will have a feature called SPAM and mark it as Yes/No. In this case the data set can be called as labeled data.
- **Unlabeled Data**
    - The data collected but with no target attributes.
    - Following above example, the data set that only has email information but the feature of whether its spam or not is not available.

### Types of Datastores
#### Databases
- Traditional relational databases
- Transactional
- Strict defined schema

### Dataware house. 
- Collects data from many different sources and formats (structured/unstructred) and collects into a datawarehouse. Before the data is landed there should be some sort of cleaning and processing happens before the data is stored. 
- Dataware house is used to store petabytes of data.
- Preprocessing is done before inserting into Data ware house. 
- Processing done on import (schema-on-write)
- Data is classified/stored with user in mind
- Ready to use with BI tools (query and analysis)

### Data lakes
- Data lakes store mass amount of unstructure data. 
- NO processing is done before dumping data to Data lakes.
- Data in data lake is historical or not sure what to be done with it
- No Processing is done before dumping data into data lakes.
- Processing done on export (schema-on-read)
- Many different sources and formats
- Raw data may not be ready for use

### Catogories of data
- **Categorical** 
    - Qualitative values associated with a group. 
    - Qualitative data is descriptive, non-numerical, and categorized based on qualities or characteristics (e.g., hair color: brown, occupation: teacher).
    - For example: creating a dataset with the feature as SPAM = True, or Feature such as dog breed name are categorical types of datasets, which can be grouped
- **Continuous** 
    - Quantitative values expressed as a number.
    - Quantitative data is numerical, measurable, and can be expressed in values and counts (e.g., age: 25, height: 5'8")
    - For example: House price feature in a dataset where it is predicted with number of roo, lot size etc. 
### Types of data handled by ML.
- **Text Data - Corpus Data**
    - Datasets collected from text, 
    - used in NLP LLMs, like news paper clippings, PDFs.
- **Ground Truth**
  - Pre processing Data which is observed and measured.
  - High quality Data Set which is successfully labelled and trusted as a source of truth
  - Ground Truth - Helps to source a good quality of the Data.
- **Image Data**
  - Refers to dataset with tagged images.
  - COnvert handwritten to text example: http://yann.lecun.com/exdb/mnist/
  - facial recognition. https://image-net.org/
- **Time Series Data**
  - Data that changes over time - Like stock market price.
- **Sound data**
    - Audio files

## Streaming Data for AI ML
- Static data, which is not changing. 
- Streaming Data, data that is constantly being updated or continuously being added. 
Some public data sets
kaggle.com 
UCI umass.edu 
registry.opendata.aws
Google public query.

- Now how we load the above data into AWS? 
### Static data -> simply upload it. 
### Streaming Data -> Sensor data, machinary data, or Stock market data, realestate data, click stream data.
- Kinesis is a family of tools such as Kinesis Datastream, firehose, video stream and Data analytics
#### Kinesis DataStream
- Data producers - Generates data, they are - Realtime user interaction, IOT devices, click stream or sensors 
- Once the data is produces Kinesis Data Stream can be used to stream, transfer or load the data into AWS. 
- This is performed using Shards. Shards are like a container that holds the data that we need to send into AWS.
- Once the data is contained with in the shard a consumer is used to process and analyze the data - which can be Lambda function or EC2.
- **Kinesis DataAnalytics** can also be used to run Realtime SQL query using the data from the shards. 
- Or can be used by AWS EMR to run Spark jobs, 
- The DataStream make sure the data is available for consumers to make sure that the data is available for further process or store. 
- Datastream to be used when:
    - Data needs to be processed by consumers/
    - Realtime analytics
    - Feed into other services in realtime.
    - If some actions needs to occur in the the data. 
    - Storing the data in original format is an optional. Just comsume for analytics.
    - If data is super important and need to store, it can store now upto 365 days.
- Example:
 - Process and evaluate logs immediately, Run some analytics, do transformation and store it in S3.
 - Process realtime Click stream: - Use the data from click stream, perform some analytics for suggestions. 
##### Shards
- each shards are made up of unique partition key
- Every time a streaming data is send it creates a sequence associated with a shard, the more request is made the more shards are created. 
- Consider two trains going in same location,
 - The train is the partition keys 
 - each cars in train is the sequence 
 - passengers are the data (Data blob upto 1 mb). 
- The default limit is 500, but can be increased. 
- Data store is transient and with a default retention for data records of 24 hours and can be increased now upto 365 days - 8760hours.

Interact with Kinesis Data Stream 
---------------------------------
Kinesis Producer Library KPL 
 - This is an abstraction to the low level APIs used in Kinesis API. Adds few features but can incur additional processing delays.
 - KPL's are java wrappers and requires java installed
Kinesis Client Library - use to abstract the low level APIs used in Kinesis API
Kinesis API (AWS SDK) - Low level API, with no delay in processing, if the data set needs to be available immediately should be used.

#### Kinesis Data Firehose
- Data producers make data - and Firehose can help to land the data directly to storage like S3. 
- Firehose can deliver to RedShift s3 splunk and elastic search.
-> Data producer -> Firehose -> S3 bucket -> Event -> Lambda -> perform some ETL and load data to -> RedShift
- Kinesis Data stream had shards and data retention but firehose dont have the shards. 
- As a reason Firehose is mostly used when you need to make sure the data is stored in S3 or some other AWS Storage location.
- Use case:
 - Save click stream on S3
 - Processing the data is optional.
 - Stream and store data from Data producers
 
#### Kinesis Video Stream
- Stream videos, stream live streams.
- Data producers sends video to AWS -> Data consumers (Like ec2 batch consumers/EC2 continous consumers )
    - Consumers gets data in fragments and frames 
- When need to process video, audio, images etc 
- Perform batch process and store the video
- feed the vedio,image and audio data into other aws service

#### Kinesis Data analytics
- Continually read and process the streaming data in realtime using SQL Queris
- Gets data from KInesis Data stream and firehose 
- then Kinesis Data analytics runs SQL queries to perform ETL and store data into S3/RedShift
- Can be used when: 
    - Have to run SQL queries on streaming data, perform some ETL and store the output. 
    - Having to create an application that depends on the streaming data, so the data can be manipulated to match with the requirement of the application.
    - Responsive realtime analytics, Create metrics, and send notification if the metrics hits a threshold.
    - Stream the ETL Jobs, make sure the sensor data is cleaned, enriched. organized and transformed before it loads to the final destination



### Which Kinesis Service to Use
| Task at Hand | Which Kinesis Service to Use | Why? |
|-|-|-|
| Need to stream Apache log files directly from EC2 instances and store them into Redshift | Kinesis Firehose | Firehose is for easily streaming data directly to a final destination. First the data is loaded into S3, then copied into Redshift. |
| Need to stream live video coverage of a sporting event to distribute to customers in near real-time | Kinesis Video Streams | Kinesis Video Streams processes real-time streaming video data (audio, images, radar) and can be fed into other AWS services. |
| Need to transform real-time streaming data and immediately feed into a custom ML application | Kinesis Streams | Kinesis Streams allows for streaming huge amounts of data, process/transform it, and then store it or feed into custom applications or other AWS services. |
| Need to query real-time data, create metric graphs, and store output into S3 | Kinesis Analytics | Kinesis Analytics gives you the ability to run SQL queries on streaming data, then store or feed the output into other AWS services. |

## AWS Services to store Data.
- S3
- RDS
- DynamoDB (No SQL - unstructured/semi structured)
    - Contains a table, inside a table you have an item, inside the item you got a key value pair. The Key value pair is called attributes.
- ReShift (Dataware house solution, relational, non relationl, structured and semi structured data)
- Timestream (the timeseries database)
- MongoDB (structured or unstructured data. It uses a JSON-like format to store documents.)

## AWS Services helper tools
- EMR - helps ot store petabytes of data in distributed system. 
- Athena - Serverless platform to run SQL Query S3 data.

Collect data -> Prepare Data -> store it in S3 so that we can use the data for ML


## References for sample data
- https://randomuser.me/

## Labs
1. Create kinesis stream.
2. Pull data from https://randomuser.me/ -> and put the data into kinesis stream. Use an EC2 instance or lambda to execute the code. 

```python
import requests
import boto3
import uuid
import time
import random
import json

client = boto3.client('kinesis', region_name='<INSERT_YOUR_REGION>')
partition_key = str(uuid.uuid4())

while True:
    r = requests.get('https://randomuser.me/api/?exc=login')
    data = json.dumps(r.json())
    client.put_record(
        StreamName='<INSERT_YOUR_STREAM_NAME>',
        Data=data,
        PartitionKey=partition_key)
    time.sleep(random.uniform(0, 1))
```
3. Once the data is ingested, create kinesis analytics using the data source as kinesis stream (When discover schema while configuring analytics you would see the schema generated). 
4. Create SQL and use below sql 
```
CREATE OR REPLACE STREAM "DESTINATION_USER_DATA" (
    first VARCHAR(16), 
    last VARCHAR(16), 
    age INTEGER, 
    gender VARCHAR(16), 
    latitude FLOAT, 
    longitude FLOAT
);
CREATE OR REPLACE PUMP "STREAM_PUMP" AS INSERT INTO "DESTINATION_USER_DATA"

SELECT STREAM "first", "last", "age", "gender", "latitude", "longitude"
FROM "SOURCE_SQL_STREAM_001"
WHERE "age" >= 21;
```
5. Save and execute you would see the subset of data from the data stream
6. Finally create the delivery stream to save the converted data. Alternatively do the transformation part by using Transform source records with AWS Lambda.
Sample Lambda function below:
```
'use strict';
console.log('Loading function');

exports.handler = (event, context, callback) => {
    /* Process the list of records and transform them */
    
    let buff = Buffer.from('\n');
    let base64data = buff.toString('base64');
    
    const output = event.records.map((record) => ({
        /* This transformation is the "identity" transformation, the data is left intact */
        recordId: record.recordId,
        result: 'Ok',
        data: record.data + base64data,
    }));
    
    console.log(`Processing completed.  Successful records ${output.length}.`);
    callback(null, { records: output });
};
```
7. Create the delivery destination, example S3 to complete the configuration.

## LAB2
- [Lambda Data Producer] -> [Kinesis Data Stream] -> [Lambda Data Consumer] -> [deliver the data into S3 using kinesis data firehose]
- Lambda function to Insert record into kinesis data stream
> A simple one
```
import json
import boto3
def lambda_handler(event, context):
    client = boto3.client('kinesis')
    data = {
        "id": "0",
        "latitidue": "0",
        "longitutde": "0"
    }
    response = client.put_record(
        StreamName = "telemetricsstream",
        PartitionKey = "geolocation",
        Data=json.dumps(data)
        )
    return response
```

A random data producer that runs until lambda times out:
```
import json
import boto3
import uuid
import random
import time
def lambda_handler(event, context):
    client = boto3.client('kinesis')
    while True:
        data = {
            "id": str(uuid.uuid4()),
            "latitidue": random.uniform(-90, 90),
            "longitutde": random.uniform(0, 180)
        }
        response = client.put_record(
            StreamName = "telemetricsstream",
            PartitionKey = "geolocation",
            Data=json.dumps(data)
            )
        print(response)
        time.sleep(random.random())
```

The consumer Lambda code
```
import json
import base64
def lambda_handler(event, context):
    #print(json.dumps(event))
    records = []
    for record in event["Records"]:
        data = base64.b64decode(record["kinesis"]["data"]).decode()
        records.append(json.loads(data))
    output = {
        "count": str(len(records)),
        "data": records
    }
    print(json.dumps(output))
```
## Data preparation
How to clean and prepare data for ML
 Process of transforming a dataset using different techniques to prepare it for Model training and testing.
Concepts:
 - ML likes clean data. 
 - Understand the data and use different techniques to prepare the data.
 - remove incomplete, incorrectly formatted, irrelevant, incorrect, duplicate data
 - The most time consuming part in ML.

#### Categorical Encoding 
- Converting categorical values into numerical values using mappings and one hot techniques 

#### Feature Engineering
- Transforming features so they are ready for ML algorithms. Ensures the relevent features are used for the problem at hand. 
- Transforming attributes within our data to make them more useful withiun our model for the problem to handle. Feature engineering is compared to an art.

#### Handling missing Values
- Removing incomplete, incorrect formatted, irrelevant or duplicated data.

Example: of the problem statement. 
| ID | Name | Evil | Affiliation |
|----|------|------|--------------|
| 1  | Luke | No   | Rebels      |
| 2  | Leia | NULL | REB         |
| 3  | Han  | 0    | Rebels      |
| 4  | Vadar| 1    | Empire      |
| 5  | Han  | 0    | Rebels      |
| 6  | Jabba the Hutt | 1 |        |
| 7  | Greedo | 0   | Bounty Hunter |

Lets assume the above dataset and its attributes. 
 - The formatting is different for few observations in the Evil attributes. You got numbers and text. 
 - In Affiliation, there are Rebels and RED
 - The missing values at "Affiliation"
 - The duplicate observations to be removed.
If the choosen ML doesn't accept text based of categorical values, then we got to find an option to encode the data into numbers. 

### Tools to prepare the data? 
- Sagemaker (Ad Hoc)
- Jupyter Notebook (Ad Hoc)
- ETL Jobs in AWS Glue (Re-usable)

### Preparing the Data
#### Categorical Encoding
- Computers are good at reading numbers rather than text data
- categorical encoding is the process of manipulating categorical variables when ML algorithms expect numerical values as input.
- Categorical variable = categorical feature = discrete feature are the features within the dataset that is broken into different categories. 
- When to encode? 
 - It depends on the algorithm you choose to solve a problem. The requirements of encoding and decoding comes from the algorithm 

| Problem | Algorithm | Encoding |
| --- | --- | --- |
| Predicting the price of a home | Linear Regression | Encoding necessary |
| Determine wether given text is about sports or not | Naive Bayes | Encoding not necessary |
| Detecting malignancy in radiology images | Convolutional Neural Network | Encoding necessary |

- Classifying categorical variables as Nominal and Ordinal is important as it determines the way you encode the values.  
- - Nominal - Order doesn't matter (A table of data with name and age)
- - Ordinal - Order does matter, (comparison of size of three birds Pelican > Hen > humming bird)

Lets take an example of below table:
| ID | Type | Bedrooms | Area | Pool Size | Price | Loan Approved |
|----|------|----------|------|-----------|-------|----------------|
| 1  | condo | 2        | 2432 | S         | 250555 | N              |
| 2  | house | 4        | 2988 | L         | 243566 | Y              |
| 3  | house | 3        | 1877 | N         | 125700 | N              |
| 4  | condo | 5        | 3876 | M         | 345000 | Y              |
| 5  | apartment | 2    | 1250 | N         | 120900 | Y              |

- Feature Loan approved:
    - which will be a target attributes for ML prediction needs to be envoded. 
    - here a simple binary encoding can be done, for example N = 0 and Y = 1. 
- Feature Pool Size (Ordinal):
    - This is a ordinal value as the order does matter (The size of pool effects the price), the encoding can be as follows
    - For pool size S = 5, M = 7, L =10 and N = 0
    - Value we choose for small, medium or large really depends on how the algorithm is tuned.
    - Changing values might change the desired output, so a fine tuning is required.
- Feature "Type" - (Nominal):
    - The home types are different. 
    - If we choose the same approach as feature pool size for encoding the algorithm might think that condo is less than house which is less than apartment. 
    - It will be a bad idea to encode in such way as this is a nominal value not a ordinal value. 
    - The options are to use one-hot encoding. 
    - **One Hot Encoding**
        - Transforms the nominal categorical features and creates a new features of binary columns for each observations.
        - Example:

| Type_condo | Type_house | Type_apartment |
|------------|------------|-----------------|
| 1          | 0          | 0               |
| 0          | 1          | 0               |
| 0          | 1          | 0               |
| 1          | 0          | 0               |
| 0          | 0          | 1               |

- As in above it creates a new feature called Type_condo and Type_house and mark as 1 if its true. 
    - If there are many many categories can we use the same one Hot encoding? 
    - The answer is no, it is not good for a dataset with many many catogories.
    - Grouping can be used or Mapping Rare Values to be used for such datasets with many features.


#### Text feature engineering
- Transforming text with in our data so ML algorithm can better analyze it. 
- One common text feature engineering will be Splitting text into bite size pieces.
- Corpus data: or Text data which is the dataset here. 
- Anything that can be considered as text, like newspaper, magazine texts, dialogue between two people. 
- Why we need text feature engineering or how it can be useful?
- - Take an example below text and see if we can get a useful information?
Raw Text: {"1 park street, sydney, 2000"}
 The ML algorithm looking at the above text as whole cannot be much of use or can it tell from it, but if we break it into different parts and determine different features or attributes from the text.
 Like break the white space, remove comas and punctuation we can get a table as below.

| Address | City | State | Zip |
| --- | --- | --- | --- |
| 123 Main Street | Seattle | WA | 98101 |

Now this becomes more useful for ML algorithm. 

##### Bag of Words
- Breaks up text by whitespace into single words. 
- Tokenizes raw text and creates a statisctical representation of the text. 
- Example: Raw Text: {"he is a jedi and he will save us"}
- Applying bag of word will tokenize each words in the text by the white space. 
- - output: {"he","is","a","jedi","and","he","will","save","us"}
- From this output we can come up with a table a statistics representation and will be useful for ML for further inference.
Example table:

| ID | word | count |
| --- | --- | --- |
| 1 | he | 2 |
| 2 | is | 1 |
| 3 | a | 1 |
| 4 | jedi | 1 |
| 5 | and | 1 |
| 6 | will | 1 |
| 7 | save | 1 |
| 8 | us | 1 |

##### One Gram
- It is an extension of Bag of Words which produces a groups of words of n size. 
- In other words breaks up text by white spaces into groups of words
- Example: Raw Text: {"he is a jedi and he will save us"}
- apply ngram with size on 1 is same as Bag of words.
- - output: {"he","is","a","jedi","and","he","will","save","us"}
- If the token size is 2 
- - output: {"he is","is a","a jedi","jedi and","and he","he will","will save","save us","he","is","a","jedi","and","he","will","save","us"}
- - Note: It also produces the output of ngram size = 1 at the end as you see in the above example.
- - If you are trying to find a group of words and flag it for spam you can use this techniques, example, click here, you are a winner. 
- unigram means that 1 word token
- bigram means that 2 word token
- trigram means that 3 word token

##### Orthogonal Sparse Bigram (OSB)
- Creates groups of words of size n and outputs every pair of words that includes the first word.
- Creates groups of words that always include the first word.
- Similar to bag of words, but it takes the very first word and uses a delemeter for the white space in the token
- Breakes token into two words and spreads them with underscore or any other delemeter and two words procuded are independed of other.
- Example: Raw Text: {"he is a jedi and he will save us"} 
- Lets use the sliding window of 4 
- output
{ "he_is", "he_a", "he___jedi" }
{ "is_a", "is__jedi", "is___and" }
{ "a_jedi", "a__and", "a___he" }
{ "jedi_and", "jedi__he", "jedi___will" }
{ "and_he", "and__will", "and___save" }
{ "he_will", "he__save", "he___us" }
{ "will_save", "will__us" }
{ "save_us" }

##### TF-IDF (Term Frequency - Inverse Document Frequency)
- "Term Frequency" = How frequent does a word appear
- "Inverse" = Makes common words less meaning ful 
- "Document Frequency" = Number of document in which terms occur.
- The, and will be less important words in most of the context
- Shows us the popularity of a words or words in text data by making common words like "the" or "and" less important.
- Represents how important a word or words are to a given set of text by providing appropriate weights to terms that are common and less common in the text
- Example: Lets assume you have two documents and you ran through TF-IDF
Document 1 

| word | count |
| --- | --- |
| the | 3 |
| force | 1 |
| Luke | 1 |
| Skywalker | 1 |
| a | 2 |

Document 2

| word | count |
| --- | --- |
| the | 2 |
| jedi | 1 |
| a | 1 |
| empire | 1 |

Since tokens "the" and "a" are showed up many times in the documents they are deemed as less important, compared to other token.

##### Vectorizing TF-IDF
- Convert a collection of raw documents to a matrix of TF-IDF features.
- (number of documents, number of unique ngrams)

##### Example use cases for text feature engineering techniques.

| Problem | Transformation | Why |
| --- | --- | --- |
| Matching phrases in spam emails | N-Gram | Easier to compare whole phrases like "click here now" or "you're a winner". |
| Determining the subject matter of multiple PDFs | Tf-idf Orthogonal Sparse Bigram | Filter less important words in PDFs. Find common word combinations repeated throughout PDFs. |

##### Other common feature engineering for text
- Remove Punctuation
- - Removing punctuations are good idea before performing ingram, osb or other text transformation.
- Lower case transformation
- - Allow to clean and standardize the text.

##### Cartesian Product Transformation 
- Creates a new feature from the combination of two or more text or categorical values.
- Combining sets of words together
- Example: Using below dataset

| ID | textbook | binding |
| --- | --- | --- |
| 1 | Python Data Science Handbook | Softcover |
| 2 | Visualization Analysis & Design | Hardcover |
| 3 | Machine Learning Algorithms | Softcover |

> Remove punctuations and apply Cartesian Product transformation which creates a new feature as following

| ID | cartesian_product |
| --- | --- |
| 1 | {"Python_Softcover", "Data_Softcover", "Science_Softcover", "Handbook_Softcover"} |
| 2 | {"Visualization_Hardcover", "Analysis_Hardcover", "Design_Hardcover"} |
| 3 | {"Machine_Softcover", "Learning_Softcover", Algorithms_Softcover"} |

- It took the value from textbook feature and binding feature and combined it together.

#### Dates Feature engineering
- Translating Dates into useful information
- Dates can come in different formats many ML programs can't determine much information from the different date and time objects. 
- Feature engineering can be used to extract more info from these objects
- Example: Was it a weekend, weekday, holiday, season, what was happening on this day 

Source Data

| Date |
| --- |
| 2015-06-17 |
| 2015-04-24 |
| 2015-02-12 |
| 2015-12-16 |
| 2015-03-14 |

Apply date feature engineering to create more useful dataset. With this information we can determine more important corelation.

| is_weekend | day_of_week | month | year |
| --- | --- | --- | --- |
| 0 | 2 | 6 | 2015 |
| 0 | 4 | 4 | 2015 |
| 0 | 3 | 2 | 2015 |
| 0 | 2 | 12 | 2015 |
| 1 | 5 | 3 | 2015 |

#### Summary 

| Technique | Function |
| --- | --- |
| N-Gram | Splits text by whitespace with window size n |
| Orthogonal Sparse Bigram (OSB) | Splits text by keeping first word and uses delimiter with remaining whitespaces between second word with window size n |
| Term Frequency - Inverse Document Frequency (tf-idf) | Helps us determine how important words are within multiple documents |
| Removing Punctuation | Removes punctuation from text |
| Lowercase | Lowercase all the words in the text |
| Cartesian Product | Combines words together to create new feature |
| Feature Engineering Dates | Extracts information from dates and creates new features |


#### Numeric Feature Engineering
- Transforming numeric values within our data so ML can better analyze them. 
- Changing numeric values in our dataset so they are easier to work. 
- Why would we need to transform?
- - Sometime ML have trouble in understanding or handling large numbers.
- - Sometimes it is better for to interpret if they group together. 
##### Feature Scaling 
- Changing numeric values so all values are on the same scale. 
- Takes large numbers and scale it to normal numbers. Reduces large calculations.
**Normalization** 
- Scaling is same thing as Feature scaling which is same as Normalization
- Example:
Lets take an example of below table:
| ID | Type | Bedrooms | Area | Pool Size | Price | Loan Approved |
|----|------|----------|------|-----------|-------|----------------|
| 1  | condo | 2        | 2432 | S         | 250555 | N              |
| 2  | house | 4        | 2988 | L         | 243566 | Y              |
| 3  | house | 3        | 1877 | N         | 125700 | N              |
| 4  | condo | 5        | 3876 | M         | 345000 | Y              |
| 5  | apartment | 2    | 1250 | N         | 120900 | Y              |

How do we apply normalization to scale down the feature Price in above table? 

Normalization makes the smallest value to zero and largest to 1 and uses the maths calculation to scale the other values between zero to 1.
Calculation:
```
         x - min(x)
x' = ------------------
     max(x) - min(x)

     125700 - 120900
x' = ------------------ = 0.021419
     345000 - 120900
```
Output value

| Price_Scaled|
|-------------|
|   0.578559  |
|   0.547372  |
|   0.021419  |
|   1.0000    |
|   0.0000    |

- This is the most common and easiest techniques but Outliers can throw off normalization. 

**Standardization**
- It puts the average price on zero and uses the z-score for the reminder of the values,.
- The zscore is calculated using the following
```
     x - Mean Price
Z = ----------------
     Price Standard
```
- It will help to smooth out the Outliers.
- average value as zero and other values are calculate using z-score.

- Feature scaling is used for many algorithms like linear, non linear regressions, neural networks and more.
- Scaling feature depends on algorithm you use.
- Normalization is on scale zero to 1. 
- Standardization has the average value set on Zero and other values are calculated with zscore. 


##### Binning 
- Changes numeric values into groups or buckets of similar values. 
- Takes numerical values which has no correlation with what we are going to find, in such case we can use binning.
- An example use case will be age of person, in which We are not concerned with exact age of a person but the age group, like 30 - 35, 35 - 50 and 50 and above.
- Using this method the age of persion is categorized and can be used as catgorical variables. 
- Binning is used to **Reduce errors between small increments of Data**

- **Quantile Binning** aims to assign the same number of features to each bin.
- Equal parts into groups is called Quantile binnings. 
- Equally group values 
- This is by creating a group based on the dataset so that each groups will have equal number of features.
- Example: re-arranging the age group into new group set where we can have equal number of users. like young, young adult, and adult.
-  
| age |
|-----|
| 25 |
| 45 |
| 18 |
| 32 |
| 40 |
| 76 |

After applying age Quantile binning 

| age | 
|-----|
| Youth |
| Adult |
| Youth |
| Young Adult |
| Adult |
| Adult | 

##### Summary
| Technique | Function |
| --- | --- |
| Normalization | From 0 to 1<br>0 - minimum value<br>1 - maximum value |
| Standardization | 0 is the average<br>value is the z-score |
| Quantile Binning | creates equal number of bins |


#### Image Feature engineering
- Extracting useful information from images before using them with ML 
- Transforming images to find useful information
- Human brain is great in understanding an image, from past experience.
- Computer lack the experience and all it has the information we present to understand the image.
- Example: One simple method
    - Assume an image with a digit "Three" in black ink on a white background.
    - The data set can be created for the entire image with a grid, and whereever the letter is in the grid can be marked as 1 and others as zero.
    - MNIST data is an example online dataset for such image comparison.

#### Audio Feature engineering
- An audio stream can be converted into numberic format against time and amplitude
Example data set: 

| Time | Amplitude |
| --- | --- |
| 0:01 | 4 |
| 0:03 | 2 |
| 0:09 | -4 |
| 0:12 | 1 |
| 0:15 | -2 |

### Data formats
- File
    - Loads data from S3 directly onto the training instance volumes.
    - Example: CSV, Parquet, JSON, Image files
- Pipe
    - Which allows to stream data directly from S3
    - Fastrer start time for training jobs and algorithms for better throughput. 
    - recordIO-protobuf (creates tensor) - creates a multi dimensional array
- Converting from file to recordIO-protobuf  can be done using below python script
```python
from sagemaker.amazon.common import write_numpy_to_dense_tensor
import io
import boto3

bucket = 'bucket-name' # Use the name of your S3 bucket here
data_key = 'kmeans_lowlevel_example/data'
data_location = 's3://{}/{}'.format(bucket, data_key)

# Convert the training data into the format required by the algorithm
buf = io.BytesIO()
write_numpy_to_dense_tensor(buf, train_set[0], train_set[1])
buf.seek(0)

# Location to upload to recordIO-protobuf data
boto3.resource('s3').Bucket(bucket).Object(data_key).upload_fileobj(buf)
```
- Feature engineering is like a layered cake. Usually there are multiple layers of transformation done to prepare the dataset.

### Handling missing values
- Missing values in dataset can make ML algorithm to do bad inference as a reason Handing missing data is an important step.
- Missing data can be represented in many ways, like - null, NA, NaN, None, etc
- First question to ask is Why are the values missing in the first place. 
    - Understand the dataset in depth will help one to find the reason for missing values
- Explain missing data using below techniques. 
    - Missing at Random
    - Missing Completely at Random
    - Missing not at Random
- Based on missing the data and its values choose one of the techniques. 
- Handle missing values

| Technique | Why this works | Ease of Use |
| --- | --- | --- |
| Supervised learning | Predicts missing values based on the values of other features | Most difficult, can yield best results |
| Mean | The average value | Quick and easy, results can vary |
| Median | Orders values then choses value in the middle | Quick and easy, results can vary |
| Mode | Most common value | Quick and easy, results can vary |
| Dropping rows | Removes missing values | Easiest but can dramatically change datasets |

Any statistical method called Mean, Median, Mode helps in replacing the missing data, which is also called as data imputation.
Supervised learning is the most difficult but gets the best results.

### Feature selection
- Selecting the most relevant features from your data to prevent over complicating the analysis, resolving potential inaccuracies, and remove irrelevant feature or repeated information
- Decide what to keep and what to get rid of
- you need to know the data well
- Example

| ID | age | fur_color | born_month | born_month_num | breed | trick_learned |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | brown | July | 7 | terrier | 0 |
| 2 | 4 | white | August | 8 | shepard | 1 |
| 3 | 3 | golden | June | 6 | terrier | 1 |
| 4 | 7 | brown | September | 8 | collie | 1 |
| 5 | 2 | black | June | 6 | retriever | 0 |
| 6 | 6 | black | May | 5 | collie | 1 |

The above dataset has the prediction for learning the trick, however determine which features really are required to perform the inference?
- The fur_color is irrelavent
- Born_month, born_month_Num are repeated and can be removed.
- The deciding fact here is the breed and age.

#### Principal component Analysis (PCA)
- It is similar to fearture selection but an unsupervised learning algorithm that reduces the number of features while still retaining as much as information as possible.
- Helps to reduce the total number of features in a dataset
- PCA isn't humans do, rather that we feed into computer during model selection
- If we had a dataset with 100s of features using techniques like PCA upfront can cut down the number of features
    - it find lenear combination of feature and removes the test.
    - Allows to better analyze the process the data
    - The less feature sets the more faster the analysis it can perform.

#### Summary 

| Problem | Technique | Why |
| --- | --- | --- |
| Data is too large due to the large number of features | Principal Component Analysis (PCA) | Algorithm that reduces total number of features |
| Useless features that do not help solve ML problem | Feature Selection | Remove features that do not help solve the problem |


### AWS Data Preparation helper tools.
- AWS Glue
    - Jobs that transforms the data on demand, schedule or on trigger.
    - One stop shop for ETL service
    - Input Data source [S3, DDB, RDS, RedShift, Other DB] -> Crawler -> Creates Data Catalog -> Use Python code to do Data preparation techniques -> Output to S3 or other service.
    - Spark job, is managed Apache Spark server.
    - Python Shell, allows to create your own python scripts.
    - Notebook for analyze and prepare Data is available in the Glue
- Sagemaker 
    - Jupyter notebook is available in Sagemaker for Data prep and analysis
- EMR
    - EMR is fully managed hadoop system, will help to pick and choose different systems.
    - Offers different framework for data preparation.
    - ETL process on EMR, then integrate sagemaker SDK into EMR to utilize the data. 
    - Apache spark can be configured to integrate with sagemaker SDK.
- Athena
    - Run SQL on S3 data
- Data pipeline
    - Process and move data between different aws services.
#### Summary

| Datasource | Data Preparation Tool | Why |
| --- | --- | --- |
| S3, Redshift, RDS, DynamoDB, On Premise DB | AWS Glue | Use Python or Scala to transform data and output data into S3 |
| S3 | Athena | Query data and output results into S3 |
| EMR | PySpark/Hive in EMR | Transform petabytes of distributed data and output data into S3 |
| RDS, EMR, DynamoDB, Redshift | Data Pipeline | Setup EC2 instances to transform data and output data into S3 |
















# AI
- AI in reality is in place from mid 60s, like spam filters, AVs 
- The compute power needed to make the AI to work was not available before 90s, but by 2011s advancement of computer power the AI is possible.
- AI is very broad object.
- - ML is a subset of AI 
  - - Deep learning is a subset of ML. 
# ML 
- ML is a subset of AI 
- Let us assume we have a data from a recent expedition (Relation between height and weight of penguine), the each data point is plotted in the graph. 
  - We got height and weight and put it up against the graph and created a pattern, now next time if we get one data of height we can predict what the weight  will be , just an example. Looking for a trend line. Which is called the training data point. 
  - a meadian line drawn in between (Linear regression) the data point might be better predicting the height/weight in this case, as the difference of acutal and redicted values are low. Where in the curved line drawn to match the point, the difference will be more with actual vs inference which is called overfitting.
##### Types of Models
- We can create a model using the data point, Models are:
  - **The Linear Regression** (Collecting the data of height and weight (X/Y axis) and creating a more predictable linear line ). This is a traditional ML argorith,.. used on basic business cases.
  - **The Logistic Regression**(Created for Yes or No scenarios), a binary output.
  - **Support Vector machine**. (Cluster information into categories)
  - **Decision Trees** (Decisioning flow) 
##### Train the model
The AI is built by choosing one of the model algorithm Linear, Support Vector or Decission tree and make a prediction. Now inorder to do that, you need to train the model with Data and train the model. By training it try to find the best values for the prediction. The output of the training is the model itself with that you can make the predictions. This is how the machine learning works.

##### How ML Works
- Every models start with the data, and choose an algorithm (What prediction that you need to make (example Use linear regression)), then take the data and alogirthm to train a mode. The training is computational expensive process. The output of this training is the model itself. Now we use this model using outside or new data to make predictions. 
- Adding extra dimention to the data, or more attributes to consider like, feather density, beak types, feet length.. We will endup drawing a multi dimensions of graph, The relationship makes it bit complicated for organic brain. But ML can do calculations using multi dimensional or all dimenstional large datasets. 

#### Deep Learning 
- Actually based on principle of organic brain, DataScientist has looked at the way the organic brain works and taken the inputs into the Deeplearning. The way Brain Neurons works with neural networks. The data scientist created *Ariticial Neurons* which takes input and outputs, and inside the neuron it has activation functions. 
- - it takes the input, process it and passes the outputs. 
- - Ariticial neural network: made up of many artifical neurons interconnected. 
- - An artificial neuron has a input and output. Inside the Artificial neuron twe have activation function, which is a function of code that takes input and process the output.
- - Having the 100s of Artifical neurons interconnected each other in a network, is the structure of Deep learning. 
- - in the same way the organic neron passes the signal, the artificial neurons passes the inputs, and configure how the input has to be processed and there is a output layer to determine the outcome. 
- - Example: If an image has to be distinguished wheather it is a Kiwi or Penguin, it needs an output called Penguin or Kiwi. The input layer passes the input of image, processed by the middle layer and gets the correct output.
- This model, comes to data requirements, time and cost.
- Labelled data takes time to generate. 
##### Neural networks: evaluation and optimization


#### Machine Learning concepts
##### Lifecycle 
1. **Collect Data** 
    - Any kind of raw data we collect, Like from filed measuring penguins, image data, machine data 
2. **Process Data** 
    - The data that we collected from the filed will have all forms of data. 
    - Get the data into a single format where the algorithm understands.
    - Then organize the data. (Creating data in to a table)
    - **Feature and Label**
      - Add features and labels into the data. 
      - Feature is the data that we use to characterise the item in the list (Example data with column, age, height, weight)
      - Label is the thing that we predict. (Example data with colum, likes the girl/boy)
        - Using above feature + label can be used to train a model to make a prediction matching the label in future. 
    - **Feature Engineering**
      - We can manupulate the feature to make better use of preduction. 
    - **Feature reduction**
      - Reduce the amount of extranous data.
      - Only use the data required to make the preduction
    - **Encoding**
      - Replace the string with a integer value, kind of masking so that the data can be referred in later stage.
    - **Formatting**
      - File format that you use to feed to the model
3. **Split the Data**
    - *Training Data -80% of data*
      - This is where we create the model.
      - How to choose the parameter and algorithm to train the data 
      - This is the reason why splitting the data is important. The training data directly influences the model.
    - *validation Data*
      - Training mechanism uses the validation data to see how acurate the model is generating. 
      - To check how well the model is doing using the data.
    - *Testing data*
      - Testing Data is not used any time during the training.
      - How well the model fits, checking underfit or overfit.
4. **Train a Model**
    The AI is built by choosing one of the model algorithm Linear, Support Vector or Decission tree and make a prediction. Now inorder to do that, you need to train the model with Data and train the model. By training it try to find the best values for the prediction. The output of the training is the model itself with that you can make the predictions. This is how the machine learning works.
5. **Test the model**
    - The data that collected during the Training data will be used to see how well the model fits. 
6. **Deploy the model**
    - Deploying model is very much of infra build.
7. **Infer (Make the predictions)**
    - Giving Real world unlabelled data, giving it to the model and asking it to label.
    - This means making predictions. 
8. **Improve (How to improve the model in real time)**
    - As part of improvement the whole cycle starts from Step 3: Split Data

##### Algorithms
 How the algorithms learns themselves?
- **Supervised Learning**
  - The original labelled data is collected using ML is used to predict or infern. 
  - Supervising, Labelled data is basics of supervised learning. 
  - We have told the ML algorithm, this is what it looks like when some one likes penguin, so that it can predict when someone likes the penguins. 
  - The Linear regression method used is another way of Supervised learning. 
  - When we have samples of labelled data, to give that for the machine learning model so that it can determine the new data and make inference.
  - Given a new input X, predict the right output y
  - Given examples of stars and galaxies, identify new objects in the sky
  - Learn a model from labeled training data, then make predictions
  - Supervised: we know the correct/desired outcome (label)
  -  Subtypes: classification (predict a class) and regression (predict a numeric value)
  - Most supervised algorithms that we will see can do both
  - *Classification*
    - Predict a class label (category), discrete and unordered
    - Can be binary (e.g. spam/not spam) or multi-class (e.g. letter recognition)
    - Many classifiers can return a confidence per class
    - The predictions of the model yield a decision boundary separating the classes
    - Refer code 1

- **Unsupervised Learning**
  - Looking for patterns in the data, in which we dont necessarly see a pattern. 
  - Imagine we have a data set with scores in x and y axis and there is no relation for the data.
    - - Find a relation ship on data that organic brain necesarily dont know the relation. 
  - It is best used when many dimensions are available in the data. 
  - Unlabeled data, or data with unknown structure
  - Explore the structure of the data to extract information
  - Many types, we’ll just discuss two.
  - **Clustering** 
    - Organize information into meaningful subgroups (clusters)
    - Objects in cluster share certain degree of similarity (and dissimilarity to other clusters)
    - Example: distinguish different types of customers
  - **Dimensionality reduction**
    - Data can be very high-dimensional and difficult to understand, learn from, store,…
    - Dimensionality reduction can compress the data into fewer dimensions, while retaining most of the information
    - Contrary to feature selection, the new features lose their (original) meaning
    - The new representation can be a lot easier to model (and visualize)
    - Used in unsupervised machine learning type.


- **Semi Supervised Learning**
  - learn a model from (few) labeled and (many) unlabeled examples

- **Reinforcement Learning**
  - It is used in robotics and automation. 
  - For example, use reward machanism, to train the model. For example, a robot is asked to pickup pengiun and if it picks wrong one we give a negative reward. The model try to get more reward. 
  - AWS Deepracer uses reinforcement learning. 
  - Develop an agent that improves its performance based on interactions with the environment
  - Example: games like Chess, Go,…
  - Search a (large) space of actions and states
  - Reward function defines how well a (series of) actions works
  - Learn a series of actions (policy) that maximizes reward through exploration

# Choosing an Algorithm
Choosing the most appropriate machine learning algorithm is vital for building high quality models. The best algorithm depends on a number of key factors:
  - The size and type of data available for training. Is it a small or large dataset? What kinds of features does it contain - images, text, numerical?
  - The available computing resources. Algorithms differ in their computational complexity. Simple linear models train faster than deep neural networks.
  - The specific problem we want to solve. Are we doing classification, regression, clustering, or something more complex?
  - Any special requirements like the need for interpretability. Linear models are more interpretable than black-box methods.
  - The desired accuracy/performance. Some algorithms simply perform better than others on certain tasks. 

# How ML Algorithm Learns?
## Learning = Representation + evaluation + optimization
All machine learning algorithms consist of 3 components:

- **Representation**: 
  A model must be represented in a formal language that the computer can handle
  Defines the ‘concepts’ it can learn, the hypothesis space
  E.g. a decision tree, neural network, set of annotated data points
- **Evaluation**: 
  An internal way to choose one hypothesis over the other Objective function, scoring function, loss function
  E.g. Difference between correct output and predictions

- **Optimization**: 
  An efficient way to search the hypothesis space
  Start from simple hypothesis, extend (relax) if it doesn’t fit the data
  Start with initial set of model parameters, gradually refine them
  Many methods, differing in speed of learning, number of optima,…
A powerful/flexible model is only useful if it can also be optimized efficiently

##### Optimization of Algorithms
- In a linear regression which line fits the model best? 
- - Measure the distance between the model and actual data. Is the prediction positive or negative? 
It uses Gradient Descent method..
###### Gradiant decent
  - Imagine you would like to get a linear regression with the best possible method, or in other words optimal prediction line in the Pengiun's height and weight Map. How do we do it?
  - We can measure difference between the points and line, and square the value and find the difference (Sum of the square residual)
    - Sum of the sqaure residual, square the difference between the actual value and linear line, then perform a sum. 
  - This is repated until we find the least residual. The value is calculated by using Sum of residual values at Y axis against "the slope of line" on X axis to find the correct slope of line regression.
  - The minumum slop of the parabolic graph is calculated with gradient of the descent. It is looking for the minimum slop or close to zero. It performs a step by step calculation of gradiant and finds the bottom of the graph or minimum value of the gradiant that is either flat or close to zero.The step size has to be right sized for the algorithm, as each step size determines how long the calculation runs, If its too short will take time. 
  - For a double dip parabolic graph botton line, it will have a different outcome, as the initial dip might not be a correct one, hence it requires a trail and error to find the best outcome. This is one of the reason the ML alogrithms are trail and errored to find the best.

###### Regularization 
- Consider the linear graph of Height and Weight of Penguin, when it comes to real world data it might not fit very well. And we apply regularization to fit the generalized data better. 
- Regularization is the type of parameter tuning to increase the generalization of model to fit the unseen data.
- We dont add a regularization to the model, unless it is found not doing correct. 
Types of Regularization
  - L1 Lasso Regularization 
  - L2 Ridge regularization
We apply regularization when the model is overfit. 

###### Hyperparameters
- Are external parameters, we can set when initiating the training Job. 
- Paramters are internal to the algorithms but Hperperameters are external. 
- Three different types of Hyperparameters
Learning rate, Epochs and Batch size. 
- **Learning rates**
  - Remember the Gradiant descent in which we learned about calculating the descent for line or parabolic graphs?
  - Determies the size of the step taken during gradient descent optimization
  - Set between Zero and One. 
- **Batch size**
  - Number of samples used to train at any one time. 
  - It could be all of the data, or some of the data. 
  - It depends on the size of infrastructure you have. 
  - If you have multiple servers, make sure the mini batch size is set and it spreads the loads across the multiple servers
- **Epochs** 
  - Are the number of times the algorithm will process the training data. 
  - Each time the algorithm passes through the data the intension is that algorithm creates more acurate model. 
  - Each epoch has one or more batches. 

###### Cross Validation
- When we split the data the validation data is optional. 
- Imagine we have a dataset, we have it split for training data, validation data and testing. 
- The validation data is used by model to tweak the hyperparameters.
- We split the data into training and testing data, and with cross validation we split the training data into multiple validation data and use it for training. All the training data is used for training and validation. This is called cross validation. 

##### Overfitting and Underfitting
- It’s easy to build a complex model that is 100% accurate on the training data, but very bad on new data
- **Overfitting**: building a model that is too complex for the amount of data you have
  - You model peculiarities in your training data (noise, biases,…)
  - Solve by making model simpler (regularization), or getting more data
  - Most algorithms have hyperparameters that allow regularization
- **Underfitting**: building a model that is too simple given the complexity of the data
  - Use a more complex model
- There are techniques for detecting overfitting (e.g. bias-variance analysis). More about that later
- You can build ensembles of many models to overcome both underfitting and overfitting
- There is often a sweet spot that you need to find by optimizing the choice of algorithms and hyperparameters, or using more data.
  Example: regression using polynomial functions

### Better data representations, better models
- Algorithm needs to correctly transform the inputs to the right outputs
- A lot depends on how we present the data to the algorithm
- Transform data to better representation (a.k.a. encoding or embedding)
- Can be done end-to-end (e.g. deep learning) or by first ‘preprocessing’ the data (e.g. feature selection/generation)

## Building machine learning systems

A typical machine learning system has multiple components, which we will cover in upcoming lectures:

- **Preprocessing**: 
  - Raw data is rarely ideal for learning
  - Feature scaling: bring values in same range
  - Encoding: make categorical features numeric
  - Discretization: make numeric features categorical
  - Label imbalance correction (e.g. downsampling)
  - Feature selection: remove uninteresting/correlated features
  - Dimensionality reduction can also make data easier to learn
  - Using pre-learned embeddings (e.g. word-to-vector, image-to-vector)

- **Learning and evaluation**
  - Every algorithm has its own biases
  - No single algorithm is always best
  - Model selection compares and selects the best models
    - Different algorithms, different hyperparameter settings
  - Split data in training, validation, and test sets

- **Prediction**
  - Final optimized model can be used for prediction
  - Expected performance is performance measured on independent test set

Together they form a workflow of pipeline
There exist machine learning methods to automatically build and tune these pipelines
You need to optimize pipelines continuously
Concept drift: the phenomenon you are modelling can change over time
Feedback: your model’s predictions may change future data

## Model building process. 
1. It starts with the business problem. 
2. Forming the machine learning problem from the business problem. 
  - Do we have all the data needed or have right algorithm to answer the busuness question. Like use Supervised, unsupervised and Reinforced learning. 
  - Then clasify the data, Binary (True or false) or Multiclass (more than two classes)
  - Observations: 
  - Labels: (Which is the future output)
3. Develop Data Set
  - The Data to be Collected and integration:
    - Structured Data 
      - With clear tables, row and columns. 
    - Semi Structured Data
      - CSV, JSON
    - Unstructured.
      - App logs, Text music and video
4. Due to various data strucute the Data has to be prepared to be an input format before it can feed into the ML program. 
    - Missing feature values can cause issue on the Mode. 
    - As a reason the missing values and outlier (value in a row is wrong) has to be removed or modified before it can feed into ML.  
    - Impute: is the way to guess and add missing value. This can be a calucalated guess, like taking a mean of values and insert it to the missing row.
5. Split data to Test validation train 
  - Generalizing the model is good thing, 
  - Model has to predict new examples accurately. For this the data is split. 
  - Cross validation techniques.
    - 1. validation, where data is split to train and test sets..
    - 2. leave on out cross validation, use only one data point for test sample and rest for training. 
    - 3. kflod - randomnly split the dataset into KFOLDs and for each folds we train the model and record the error.
6. Data visualization and Analysis.
  - **Feature** refers to any attributes choosen to be used as a data point in training data set. Example, Height and Weight of Penguins. 
  - **Label** is not a feature in the training data but label is the variable that will be predicted by the Model. 
  - Using the visualization we can understand the properties of data better. 
  - We can use Statistics, Scatter-plots and Histograms for Visualization and Analysis.
7. Feature engineering:
  - It is the process of manupulating the raw or original data into more useful features. 
  - It requires lot of trial and error. 
  - What I am using to make my decission. Is there a feature that is noticed during data analysis and how can I make use of it?
  - Converts raw data into higher represenation of data. 
  - Binning (grouping the dataset) helps to convert a linerar dataset to non linear for the model. For example lets assume salary data that has a feature called age and salary for age group are same for 20-30, same for 30-40 and so on for 40-50, binning them together makes it easy to make it non lenear and showcase the relationship. 
8. Train the model.
  - Train the model multiple times based on variables are called parameters. 
  - Parameter tuning is performed during the training to improve the performance. 
  - Parameters are the nobs to improve the ML inferences.
  - Loss Function: That calculates how dat the predictions are from the ground truht values. 
  - Regularilization: Increase the generalization of model to fit the unseen data. 
  - Learning Paramters (decay rates): Controls how fast and slow the model learns. 
9. Evaluate using Test data:
  - How accurate the model was using the test data. 
  - Look for valuation accuracy. 
  - Perform Root mean square error RMSE of the model using validation data.
  - Perform Mean Absolute Percent Error of the model using validation data.
  - The low the value of RMSE the better the model. 
10. Evaluate the model
  - During evaluation the data should be fit to generalize the unseen data. We should not fit the training data to fit the max accuracy but the test data validation accuracy for evaluation.Using training accuracy is called overfitting...
  - If not enough feature to train the model it will lead the model not to generalize as not enough data and is called underfitting. 
  - Evaluate based on the business success criteria. 
  - If we need more data or better data to prevent overfitting we can add Data Augmentation and Feature Augmention to the pipe line. 
The goal is to deply a model into production:
  - To predict the model on production the prod data should have same features as in the testing data. 
  - Since data distribution can drift over time, deploying data model is a continuos process. It is good process to continuously monitor if prod data is deviated from training data. Or train the model periodically. 
Machine Learning start with Business problem and results in preduction. Training data set is input to the machine and evaluated by it and creates the model.
Test data sets are used to test the data set for accuracy. If predictions are not accurate the ML is taught to learn better. 


### Data science and machine learning on AWS
#### Roles of Data scientist. 
- Exploratory data scientings - Help me make sense of the data. 
  - Use sagemaker studio IDE. 
- Machine learning Engineer - Help me make and run some ML Models. 
  - Helps to deploy the tools in the production. Pipelines, Training etc. 
 They work with Devs and platform teams along with Business owner (The business outcome, like what the solution is actually going to do). 
-  

## AWS Application services - AI/ML. 
### Amazon Rekognition 
- Image and Video analysis tool. 
- Pre trained deep learning model. 
- Simple API 
- Image moderation 
- facial analysis 
- celebrity recognition 
- face comparision 
- Text in image. 
 Can create filter to prevent inapproporiate images, Extract text, scan image library to detect famous people, how many are smiling etc. 

#### Reckognition Video 
- Stored video analysis.
- Streaming live videos. 
  Can be used to detect the people from a captured security footage. Detect offensive content from the video uploaded to the platform.

### AWS POLLY 
- Text to speech 
- female to male conversion. 
- deep learning powered simple API.
 Read web content, accessibility tool for people who has vision impared. 
 Automated voice response for a telephony system.

## Amazon Trabscribe
- Speech to Text conversion. 
- Custom Vocabulary to include the words that you need to be captured. 
- Transcription jobs : Can perform the job of transcribtion in batches or in job and managed by API.
- Text search from a media file, or recording that we have made. Look for specific word and take action on it. 
- Also use the data to pass it own to AWS Comprehend to detect the customer sentiment. 

## Amazon Translate 
- The text translation tool from one language to another. 
- Enhance online chat application to translate conversation in real time. 
- Batch translate documents 
- Publish news and solutions to multiple languates. 

## Amazon Comprehend
- A text analysis tool, a NLP too, that uses pre trained deep learning. 
- Keyphrase extraction 
- Sentiment analysis
- Syntax analysis 
- Entity recognition (Is this text talking about a person)
- Medical named entity and relation ship extraction (NERe). Extract infromation from the medical notes 
- Custom entites 
- Language detection 
- Custom classification - We can train with our own data to get a custom outputs.
- Topic modeling. 
- Multiple Language support. 
 What is the overall sentiments of my organization from the comments posted by customers. 
 Comprehend can be trained to work with unstructured clinical data to assiust in research and analysis. 

## Amazon Lex 
- The conversation interface service like Alexa, and it uses Automatic speech recognition and Nature Language understanding. 
- It can be used to create a chatbot. 
- It can be integrated with Comprehend to understand what is the sentiment of customer who initiated the chat and respond accordingly. 

*AWS Lambda functions can be used to combine all the AWS AI ML Application services, much easily. And use the AWS Step function to orchestrate the entire process and decouple it into multiple AWS Lambda functions*

### Machine Learning for Business Leaders
- ML can help to overcome a persistent problem. 
- Data collection, improvement and ensuring the quality of data is key to the success for ML projects. 
You can enable the ML team by asking critical questions.
- What are the assumptions made?
- What is the learning target (hypothesis) ?
- What type of ML problem the team is trying to solve?
- Why did you choose this algoritm?
- How will you evaluate the model performance?
- How confident are you that you can generalize the results?
*Better the data better theprediction better recommendations and more satisfied customer More sales*
ML system use past data and build the model and use to predict the future data. 
- When you should consider using machine learning to solve the problem?
  - Use machine learning when software logic is too difficult to code. 
  - Use machine learning when the manual porocess is not cost effective.
  - Use machine learning when there is ample training data available. 
  - Use machine learning when the problem is formalizable as an ML problem. 
    - formalizable - Means reducing a problem to well known ML problem. 
- When is Machine Learning NOT a Good Solution?
  - Dont use Machine learning if there is no Data. 
  - Dont use ML if there are no Labels available on Data or can't label the data. 
  - If you need to launch a product quickly then use a simpler version and lauch and wait for the feedback to invest more using ML.
  - No Tolerance for error, then avoid using ML.
- Data
  - How much data is sufficient for building a successful ML Models
  - Data Quality - How to deal with data quality issues?
  - Data preparation prior to model building. 

### Amazon Textract
- What does Amazon Textract do?
  - Amazon Textract is a machine learning (ML) service that helps you extract text, handwriting, and structured data such as tables and forms from documents. Amazon Textract goes beyond simple optical character recognition (OCR) by extracting relationships and structures between elements from documents.

### AWS Sagemaker
 - With AWS Sagemaker, you could almost spend all time with Sagemaker to get the AI ML stuff done. 
 - Provides every developer and data scientist the ability to build, train and deploy machine learning models, A complete lifecycle for ML Program. 
 - It gets you the Tools, Code, scripts, Infra management, Algorithms in managed services. 
##### The three stages of SageMaker. 
1. Build (Ground Truth & Notebook)
2. Train
3. Deploy
###### Build (Ground Truth & Notebook)
  - Pre processing Data
  - Get High quality Data Set 
  - Use Note book to manipulate the Data
  - At this stage you choose which algorithm best suits your business requirements.
  - Ground Truth - Helps to source a good quality of the Data. 
- Data pre-processing using Sagemaker Build. 
  - Business problem, We need to have a good understanding of the problem that we are going to solve before we try Machine Learning. 
  - In this stage, take a look at the data, and look forward to see if the problem that we have to solve. 
  - Data pre-processing steps (ETL). 
    - **Visualize your Data**. 
      - The sourcing of data, assume you have the data, you might need to visualize the data. For visualizing the data should be formatted in such way, For example using AWS Quicksight, or use Jupyter Notebook with in sagemaker and use Matplotlib lib to draw the graph using the data. 
      - Visualizing is used for Sanity check, to see how the data look like. 
    - **Exploring Data**.
      - Lets run some statistical analysis and set some data what we have got it. 
      - Do we need to manipulate the data? 
      - If we need to manipulate the data? Then we need next step - called feature engineering.
    - **Feature engineering**. 
      - If the data is small, then we can do it inside Notebook. 
      - If data is huge, inside data warehouse? then use AWS EMR or use Spark. 
      - Make sure the data is represented in a way that we know the ML algorithm can make best use of the data. 
    - **Synthesize the data**. 
      - Do you have more example of the data ? If not we syntesize the data.
      - Example, building ML for IOT device data, if there is not much data available, then build from the similar data. 
      - If we have a nice set of data, then we can syntesize based on that data to create more data for ML. 
    - **Convert**.
      - By this time we know what algorithm that we gonna use, and should know what format the data needs to be preented. 
      - Convert the data into the format or change the structure on how the data needs to be presented for ML algorithms. 
    - **Structure**. 
      - Here we do the data manipulation to make sure the data is getting into the structure. 
    - **Split Data**.
      - Splitting the data into Training Data, Validation Data etc. 
  Sagemaker Notebooks can use to perform all of the above activities. And Sagemaker Algorithms Can be used to explore the data as a pre-processing stage for training. 
  **Amazon Groundtruth**
    - It can help in performing labelling the data for the ML Algorithms. 
    - The Ground truth model will work in conjuction with Humans to perform the data lablelling. 
    - For example assume you want to label the data of Dog and Cat images to distinguish the image for an ML mode, then the first thing to do is to perform training of the ML model with some labelled data, by labelling the image dog and cat so that next time when we feed a unlablled data the model can perform a inference. 
    - The Ground truth works along with the human being to Label data initially using an endpoint, which has a machine learning model that can label the unballed data. This ML model can't label it by itself, but it will use the initial set of data and share it with a private set of users. The humans can label the data and send it back to the model. 
    - The Amazon Mechanical Turk, is a service around sending simple and small jobs to humans. 

###### Train (Training)
  - Use Built in Algorithms. 
  - Perform hyperparameters tuning. 
  - Managed notebook instance to control the training. 
  - AWS will deploy the infra to control the training. 
###### Deploy (Inference)
  - It can perform realtime deployment. 
  - Batches 
  - Use Notebook to perform teh deployment. 
  - Build the deployment infra 
  - Neo - To perform the deployment to edge devices. 
###### API For Sagemaker. 
- AWS API to control Sagemaker - This API can be used to interact with the service to build, train and deploy the models. 
  - Like building Sagemaker env.
- Sagemaker Python SDK - APIs can be used to desgin and perform actions within Sagemaker. 
  - Like building models, training a model and performing a inference. 

##### Sagemaker Notebooks.
- Sagemaker notebooks are jupyter notebooks in managed service. 
- Notebook instances:
  - It allows you to get into the notebook and performing the code execution. 
  - If it is used as a control center then smaller size is more than enough. You got to choose the larger instance type if you perform any local data processing. 
  - Elastic Inference : The GPU capability.
  - IAM Role: same like EC2 instance profile. 
  - You can enable/disable root access.
  - VPC, is optional if you want to run the notebook instance in vpc. 
  To open the Notebooks. 
  - Open Jupyter 
    - It will redirect to a web endpoint of Jupyter notebook running inside the AWS Account. 
    - It will create a presigned URL, using the role used to access the console. The reason is that only the authenticated role will only be able to access the link. 
    - You can take an example from one of the notebooks examples. 
  - Open Jupyter Lab
    - The second endpoint for Jupyter Lab interface.
- Lifecycle Configuration:
  - This is much like bootstrap scripts on EC2 instance. 
  - You can add scripts that can run during startup or once the notebook is up. 

#### Samples: Perform ETL using Sagemaker Notebook
- https://github.com/linuxacademy/content-aws-mls-c01/tree/master/SageMaker-Pre-process-Image-Data
#### Samples: Sagemaker Algorithms. 
  - AWS has built in algorithms available in Sagemaker. 
  - The algorithms are available from AWS Market place.
  - Using Sagemaker we can also create the custom Algorithm. 
  - The built in algorithms, few are:
    - BlazingText 
      - Word to vec/ Text classification. 
      - It is a Natural Language processing 
      - sentimental analysis 
      - The amazon comprehend is one of the example using this example.
    - Image classification algorithm
      - It is a convolutional neural network algorithm
      - It can use for image recogniztion.
      - Amazon Rekognition is an example.
    - K-means algorithms
      - Based off web-scale k-means clusterning algorithm. 
      - Find descreate groupings within data, Use the ungrouped or unlabelled data and label it. 
    - PCA Principal componenet Analysis PCA Algorithm
      -  Reduce the dmiensionality. 
      - Find the number of the feature so that you can decide on which feature that you can reduce. 
    - XGBoost Algorithm 
      - One of the leading algorithm making predictions from tabular data (Decission tree)
    
### SVM (Support Vector Machine)
 - If you have a bunch of object and you want to catagorize them into two or multiple catagorise. 
 - Say if you have a pic and want to catagarize if it is a dog or cat. 
 - It uses line on 2D and plane on 3D to catagorize the object, SVM finds the best methoid and maximize the distance between the objects called margin.
 - It is the supervised learning logirithm. 
 - It can be loaded from the python library 
 ```from sklearn import svm```
 - Easy to understand, implement, use and interpret. 
 - Kernel Trick allows to perform complex methid.
 - used for face recogniztion, spam filter, sign detection 

#### Kernel Trick (https://www.youtube.com/watch?v=Q7vT0--5VII)
- It creates non lenear transformation in complex SVM. 
- Kernel Trick  
- To separate teh data properly, using a curved discission boundary. 
-
# Discriminative AI (Predictive AI)
- A model that just predicts the outcome.

# Class room Training notes on Jan/16
# Generative AI 
### LLM 
- Large Language Model. 
- One usage is to complete the text... for example, internet search engines, typing message on phones.. etc. 
- Imagine that we are writing books on football, and training AI to assist..
  - First step will be training the AI on large amount of data on the football. 
## Transformers and self-attention.
- The exponensial numbers of word combinations that needed to predict a whole sentence needs more memory power... As a reason another approach is taken using Tranformers..
- A sentences are divided called as Tokens and considers Semantics & position and assigns a score.
- Tranformers can run this calculations in paralell.
### Transformers Architecture
- Encoder only
- encorder-decprder
- Decorder Only (Used by the Chat GPT)
## Developing LLM 
- Foundational models
  - Creating a LLM from scratch needs more resource, cost, maintanance, skill set. The Foundational models are LLMs that is already trained on large amount of data.
  - The Prompt Engineering.
  - The FMs needs prompt engineering.. 
    - Context window
    - Few shot prompting. Examples of predictions to make and use that knowledge to make predictions.
- Model fine-tuning. 
  - Taking a FM then conducting additional training of that model with domain spcific knowledge. 

# Types of ML Approaches
- Two types of predictions.
  - Clasification - Like yes or no. 
  - Regression - When you want to predict the continuous value. 
## Types of datasets and ML problem domains.
  - Data sources
    - Text source
    - Image source.
    - Tabular source. - The more standard tabular data.(data from database???). 
## ML Lifecycle
  [frame the ml problem(Role: BA)] -> [process data] -> [Model development] -> [deploy] -> [monitor]
- Frame the ML problem:
  - Frame the business problem. 
  - What the business stakeholder needs.. 
  - Example: which of their customer might leave in near future for a business so that they can take preventive action in advance. 
- Process data:
  - Use the data including billing, revews etc..
  - Example for citizen salary package program, fing a target whom we can give the package. 
    - The tabular data with the salary information on the citizen can be used. 
- A good it:
  - Overfitting
    - Example, situation, where studing the sample exam question paper and only memorize the questions and answers but on real exam you cant perform as you only know about the sample data
    - High variance and low biase
    - Difference in the ability to genearlize relation ship is called variance.
  - Underfitting
    - May be the model is too simple, or the data is not good enough.. or haven't trained the LLM well.
    - Low variance and high biase
    - inability to capture the relationshop between datapoints in alinear regression - is called bias. In underfitting is is very high as it uses the linear regression.
  - Balanced
    - It is between overfitting and underfitting. 
    - Model stays in between variance and bias
  - Prediction error:
    - 
## Bias and Variance example
Study for longer and everyone gets 70% - high Bias example (underfitting)
Study hours is longer + Color of pen + day of study + and scores very high and assumes that pattern holds true every time.. - High variance example (Overfitting)

# Preparing the Dataset
## Process Data step in above ML Lifecycle. 
  - Collect data 
    - Example Datalake: S3 bucket. Transcational , Database, Clickstream and Iot Sensors..
  - Pre-Process data
    - Needs some cleanup.. maybe the tabular data is not clean. Have to deal with missing data, incorrect, feature not consisting... Drop of the data is irrelevant. Normalize the data. 
    - Which means the data needs transformation.
  - Analysis and visualization
    - As an initial step the dataset features needs to be analuzed.  
    - Can use visual data analysis.. example histograms (represents numberical featues) and bar charts (catagories features) can be used to perform the same. 
    - Density plot and box plot both gives a visual idea of where the data is distributed.
    - Multivariance statistics: - Comparing features against others.
  - Engineer features
      - All the data visualization should be used to make dicission on what datasets features to be used on training the model. And drop them if needed.
  - Split dataset
    - The original data is divided into 3. 70% of the data is used for train the model.. 15% will be used to evaluate and finetune the mode. Final 15% is used to test the model when completes the build.
    - Training Data
    - Validation Data
    - Testing Data
    - Example, think of you got 100 questions to study for exam, you use 70% to learn, then 15% to validate your learnings and with the feedback go back and study again on the 70% of questions... then have last 15% to really test your knowledge.
## Dataframes 
-  In python It is table, that can store tabular data.

# Lab preperation
 -> Data preparation .

# Developing mdel / Training a model. 
- We have so far got the data for training...
## Chose an algorithm..
- Algorithm selection comes to what type of data you have and what type of business predection you want to solve (clasification or lenear for example)
- Sagemaker comes with built in algorithms
- An example of decission tree algorithm is using XGBoost. 
## Training the model in Amazon Sagemaker 

## Evaluating and Tuning a model.
- This steps fit into the final stage of model development.
- Bias and variants are the two components of predictions errors. We have to minimize both. 
- A confision matrix helps to visualize the outcome of number of predictions falls into the predicted vs actual metrics. 
- Acuracy:
  - Overall proportion of correct predictions 
  - Calculating the Acuracy:
  -  howmany precictions made are correct. True positive + True negative / All preductions that made.. which gives the accuracy. 
- Precision: Focus on minimizing the false postives.
  - Proportion of positive predictions that are actual positives
- Recall: 
  - Proportion of actual positives predicted positively
- F1 Score: - is the balance between precision and recall. 
### evaluation calculations
    Accuracy: A measure calculated by dividing the sum of True Positives and True Negatives by the sum of Positives and Negatives. The calculation is (TP + TN) / (P + N).
    Precision: A measure calculated by dividing the True Positives by the sum of True Positives and False Positives. The calculation is TP / (TP + FP).
    Recall: A measure calculated by dividing the the True Positives by the sum of True Positives and False Negatives. The calculation is TP / (TP + FN).
    F1 score: A measure calculated by multiplying prevision and recall, and then dividing that number by the sum of precision and recall. Then, the result is multiplied by 2. The calculation is 2 * ((precision * recall)/(precision + recall)).

### Address Classification isues
- Receiver operative curve (ROC). 
- Area underneath the curve (AUC).
# Model tunning and hyperpatameter optimiztion
- Metrics after training can be used to optimize the model/
- Hyper parameters helps to find how ML algorithms works.. how many rounds of training it has..
- using hyper parameters, we can influcence the bias and variance and finetune the model.
## Determining the hyper paramters to choose for a model
  - Use Grid search: Runs every single combination
  - Random search : Runds randomnly until it hits a threshold that is set.
  - Bayesian search: It is like a treasure hunt.. go some educated hints.. and based on previous attempts makes a guess to choose right hyper parameters.
  - AMT Amazon sagemaker automatic model tuning can tune the model automatically.
### Hyperparameters tuning considerations:
  - adjust small numbers of values

# Deploying the models and use it
- First needs to identify the inferencing requirements. 
- Choose the types of endpoints 
- Configure and create the endpoint.
- Sagemaker inference (Deployment) options to consider:
  - Realtime? 
  - Serveless? 
  - Asynchronous (near real time)
  - Batch (can afford wait time, overnight)


# Sagemaker canvas
 - A tool without need to do a code. 
 - a basic machine learning with no code can use this. 
# sagemaker studio lab
 - no need of AWS account, designed to get people hooked into sagemaker studio.
 - this is a free tool.
