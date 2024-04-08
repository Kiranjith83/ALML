### AI ML notes
| Ref: Reference: https://ml-course.github.io/master/notebooks/01%20-%20Introduction.html



# Machine Learning concepts
## Lifecycle 
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

  ##### Train the model
The AI is built by choosing one of the model algorithm Linear, Support Vector or Decission tree and make a prediction. Now inorder to do that, you need to train the model with Data and train the model. By training it try to find the best values for the prediction. The output of the training is the model itself with that you can make the predictions. This is how the machine learning works.

##### How ML Works
- Every models start with the data, and choose an algorithm (What prediction that you need to make (example Use linear regression)), then take the data and alogirthm to train a mode. The training is computational expensive process. The output of this training is the model itself. Now we use this model using outside or new data to make predictions. 
- Adding extra dimention to the data, or more attributes to consider like, feather density, beak types, feet length.. We will endup drawing a multi dimensions of graph, The relationship makes it bit complicated for organic brain. But ML can do calculations using multi dimensional or all dimenstional large datasets. 

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
- Each shard can support up to 1 MB/sec or 1,000 records/sec write throughput or up to 2 MB/sec or 2,000 records/sec read throughput.


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
    - If we choose the same approach as feature Type for encoding the algorithm might think that condo is less than house which is less than apartment. 
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
- TF - term frequency (Words frequency)
- IDF - Inverse Document frequency (Sentence Frequency )
- Helps to find which words in a sentence is more significant.
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
- How does it calculates?
- TF * IDF (higher the value the more significant)
- Calculate TF - how many times a term occur in a document. 
```
         (Number of times term (t) appears in a document)
 TF(t) = -----------------------------------------------
          (Total number of terms in the document)

Example: he is a jedi and he will save us
tf(he) = 2/9 =  0.22222
tf(jedi) = 1/9 = 0.1111111

```
- Calculate IDF - Get the weight of rare words. The words that occur rarely in the corpus have a high IDF Score

```
         log(Total number of documents or number of rows)
IDF(t) = ------------------------------------------------
          (Number of documents with term (t) in it)
Example: he is a jedi and he will save us   
         he is the exterminator and he will kill pests
         he is an ice cream man and he will save us   

IDF(he) = log 3/6 = 0.07952020911
IDF(jedi) = log 3/1 = 0.47712125472
```

- Example module to vectorize is TfidfVectorizer from sklearn
- Once vectorized you the vector is identified as below.
(The row number on which the records that belongs to, encoded value of the word)

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
- An outlier is an individual point of data that is distant from other points in the dataset. It is an anomaly in the dataset that may be caused by a range of errors in capturing, processing or manipulating data. Outliers are data points that deviate significantly from the rest of the distribution, and they can have a big impact on your feature selection process.


**Standardization**
- It puts the average price on zero and uses the z-score for the reminder of the values.
- Assume you have 83% in maths test where class mean was 81%, and standard deviation is 7.3%, the standardization will help to calculate how you performed overall compared to class with mean of zero, the ZScore in such scenario will be 0.274. Which means you are better than overall class with the margin of 0.274 points.
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

### Data prep Lab
- Assume the data is loaded into S3
- Answer business question from the data.
    - What percentage of users are female (Use Athena to run the query against the data)
- Convert to CSV 
    - Use AWS GLue job to run apache spark to convert to CSV
- Mapping gender catogorical value to numberical values
    - Use glue
Overall flow
S3 -> Glue Crawler -> Glue Catalog -> Use Athena to answer business question -> Glue Job Apache spark (Perform transformation) -> Load data to S3

1. Setup Athena catalog from Glue Catalog. (Add Crawler on Glue and use it on Athena)
2. Run SQL Query to check the data
```
SELECT COUNT(*)
FROM "my-user-database"."my_user_data_output_bucket"
```
```
-- percentage of gender 
SELECT gender, (COUNT(gender) * 100.0 / (SELECT COUNT(*) FROM <AWS_GLUE_TABLE_NAME>)) AS percent
FROM <AWS_GLUE_TABLE_NAME>
GROUP BY gender;
```
```
-- Most common ages
SELECT age, COUNT(age) AS occurances 
FROM <AWS_GLUE_TABLE_NAME>
GROUP BY age
ORDER BY occurances DESC
LIMIT 5;
```
```
 SELECT SUM(CASE WHEN age BETWEEN 21 AND 29 THEN 1 ELSE 0 END) AS "21-29",
        SUM(CASE WHEN age BETWEEN 30 AND 39 THEN 1 ELSE 0 END) AS "30-39",
        SUM(CASE WHEN age BETWEEN 40 AND 49 THEN 1 ELSE 0 END) AS "40-49",
        SUM(CASE WHEN age BETWEEN 50 AND 59 THEN 1 ELSE 0 END) AS "50-59",
        SUM(CASE WHEN age BETWEEN 60 AND 69 THEN 1 ELSE 0 END) AS "60-69",
        SUM(CASE WHEN age BETWEEN 70 AND 79 THEN 1 ELSE 0 END) AS "70-79"
 FROM <AWS_GLUE_TABLE_NAME>;
```
3. Transform data from json to CSV and change the gender attribute to a binary value ( from male/female to 0 or 1).
```
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from itertools import chain
from pyspark.sql.functions import create_map, lit
from awsglue.dynamicframe import DynamicFrame

args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session 
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

datasource0 = glueContext.create_dynamic_frame.from_catalog(database = "<AWS_GLUE_DATABASE_NAME>", table_name = "<AWS_GLUE_TABLE_NAME>", transformation_ctx = "datasource0")

# Here is the custom gender mapping transformation 
df = datasource0.toDF()
gender_dict = { 'male': 1, 'female':0 }
mapping_expr = create_map([lit(x) for x in chain(*gender_dict.items())])
df = df.withColumn('gender', mapping_expr[df['gender']])
datasource_transformed = DynamicFrame.fromDF(df, glueContext, "datasource0")

applymapping1 = ApplyMapping.apply(frame = datasource_transformed, mappings = [("first", "string", "first", "string"), ("last", "string", "last", "string"), ("age", "int", "age", "int"), ("gender", "string", "gender", "string"), ("latitude", "double", "latitude", "double"), ("longitude", "double", "longitude", "double")], transformation_ctx = "applymapping1")

datasink2 = glueContext.write_dynamic_frame.from_options(frame = applymapping1, connection_type = "s3", connection_options = {"path": "s3://<S3_BUCKET_NAME>"}, format = "csv", transformation_ctx = "datasink2")
job.commit()
```

## Data Analysis and Visualization
- So far we have fetched the data, cleaned and prepared the data, the next step prior to use the data to train a model is to Analyze and Visualize the data. 
- Cross check if the data is completely ready for training, or need to go back and prepare the data once again.
- Technical tool Bring important information about the data using visualization.

Sample data:

| car | price | date | 
| --- | --- | --- | 
| mustang | 287326 | 06/2015 |
| Ferari | 42342 | 4/2015 |
| mustang | 334534 | 2/2015 |
| suziki | 23423 | 12/2015 |
| Ferari | 52423 | 3/2019 |
| tata | 42345 | 3/2018 |
| toycar | 23425 | 3/2016 |


What is the best technique to visualize the data? 
Different categories helps one to analyze the data are :

1. **Relationships**
  Do we want to find important relationships within our data? Are there any trends or outliers?
  - Helps to provide a good general overview, show distributions and correlation between attributes. Helps to find outliers and extreme values.
  - Methods to compare relationships are -
    - **Scatter Plots** (Relationship between Two attributes )
      - Is there a relationship between size of house and its price (size on x axis and price on y axis)
    - **Bubble Plots** (Relationship between Three attributes )
      - is there a relationship between size of a home, age of the home and the price? (size on x axis and price on y axis and age can be denoted as bubble)
  - The visualization helps to identify any correlations between two or more attributes. 
    - Positive Correlation
    - Negative Correlation
    - No Correlation

2. **Comparisons**
  Are we comparing different values within the data?
  - Visualizing comparisons in your data can provide a static snapshot of how different variables compare and show how different variables change over time.
  - Methods to compare Comparisons are -
    - **Bar Charts** (Look up the status for single point in time)
      - Bar charts using bars to make single variable values comparison. Provides a way to lookup and compare the values.
      - Example: compare price attribute of cars. Shows static snapshot for the cars. Helps to understand what is the price comparison of cars at given point of time. 
    - **Line Charts** (Helps to visualize data changes over time)
      - Shows one or more variables changing over time.
      - Example: Compare price attribution of cars that changes over time. X axis with price, y axis with dates and different cars to be color coded.

3. **Distributions**
  Do we want to know more about the distributions of data? Are there any outliers?
  - How the data is grouped or clustered over certain intervals.
  - Methods to compare Comparisons are -
    - **Histograms**
      - Graphs put values into buckets or bins and determines a measurement.
      - Assume you got a car dataset with name, price and date as attributes/features. 
      - price on x axis and number of cars on y axis. We can group the number of cars in a group of price range (bins of price).
      - This helps to increase the number of bins to see further distribution of data. (as the number of cars falls into different price bins)
    - **Box Plots**
      - Graphs show a wealth of distribution information, Can see things like lowest and highest values outliers and where the most values fails.
      - Box plot shows a lot of information when visualizing distribution.
      - Assume you git y axis with price range and x axis with car names, a box plot will show a max and min value, with a median value, An extreme value of price (may be an outliers) also a box in the box plot shows the majority of cars for the price.
      - Another example is to show the distribution of test scores for a given exam.
    - **Scatter Plots** 
      - Also knows as scatter charts. Points long x and y axis for two values. Can show clustering and distribution of data
      - As same as relationship method to view but added changes over time. For example, showing scatter plots of cars prices over time for different cars.
      - Show return of investment for the money spend and the total time invested

4. **Compositions**.
  Want to know what makes up the data? What are the components of the data?
  - What are the data made of?
  - Methods to compare Comparisons are -
    - **Pie Charts**
      - Pie charts show how various values compare as a whole share of the total 
      - What percentage of car is mustang in the entire dataset.
      - Another example: Showing hte sales figures for each region
    - **Stacked Area charts** 
      - Stacked area charts shows the measurement of various items over longer period of time. 
      - Lets assume you have another feature added to the dataset called dealership where the car is sold
      - The stacked are helps to show Dates on x axis number of cars sold in y axis and the car dealer in color coded stack(lines).
      - Another example: Showing the number of products sold by different departments on a weekly basis.
    - **Stacked Column charts**
      - Stacked bar charts - show quantity of various items over shorter periods of time.
      - Lets assume you have another feature added to the dataset called dealership where the car is sold
      - The stacked are helps to show dealership name on x axis number of cars sold in y axis and the dates in color coded bars.
      - Another example: Showing the quarterly revenue totals for each region.


Different tools are available to visualize the data. 
- Developer Tools
  - Pandas
  - Python
  - Jupyter
  - matplotlib
  - Sagemaker
  - scikit learn

- Business Intelligence Tools
  - Tableau
  - Amazon QuickSight
    - BI tool to create visualization

### Overview of categories and corresponding graphs for visualization
![alt text](visualization_graphs.png "visualization_graphs")
### Choose a visualization for a problem
- Picking the right visualization method depends on what you want to see. 
 - is it Relationship?
 - is it Comparison? 
 - is it Distribution ?
 - is it Composition ? 
- **Heatmap**
  - Shows can show more than two different categories like composition, distribution and comparison. Example: Population density map for the world. 



### Lab 
- Create AWS Quicksight 
- Create boxplot using matplotlib in jupyter notebook



# Train the model
- So far, we collected data, cleaned, prepared, encoded and time to train t he model.
- Garbage in garbage out, good data in better the inference.
- First in foremost we have to have a problem to solve and then there should be data that describes the problem that we are trying to solve.
  - We can add a ML algorithm and through computation the ML will try to reverse engineer the situation by looking at data and comeup with the mathematical formula to generalize the problem, Now when you give the new problem to the model (similar to training data) the mathematical formula can infer the outcome based on what it learned and generated from the training data.

### Components in developing a good model
An ML Model will have below components.
  - Generalization (what are we trying to achieve)
  - Model
  - Data
  - Algorithm
  - Computation
  - Feedback 

- Are we forecasting the number? Decides what data you need and what algorithm that we are going to use.
- Do we really need a ML ? can we use something that was already build?
- Is prediction realtime, or can be done in batch. 
- Do we have the data to work with? Do we have enough or more data (how much feature engineering should be done)?
- How can we tell if the inference is working as expected?

In detail: 
1) What type of generalization are we seeking?
Do I need to forecast a number? Decide whether a customer is most likely to choose Option A or Option B? Detect a quality defect in a machined part?

2) Do we really need machine learning?
Can simple heuristics handle the job just as well? Can I just program some IF...THEN logic? Will a linear regression formula or a look-up function fulfill the needs?

3) How will my ML generalizations be consumed?
Do I need to return real-time results or can I process the inferences in batch? Will consumers be applications via API call or other systems which will perform additional processing on the data?

4) What do we have to work with?
What sort of data accurately and fully captures the inputs and outputs of the target generalization? Do I have enough data? Do I have too much?

5) How can I tell if the generalization is working?
What method can I use to test accuracy and effectiveness? Should my model have a higher sensitivity to false positives or false negatives? How about Accuracy, Recall and Precision?


## Different types of models and different styles of their learning

|Properties below | Supervised Learning | Unsupervised Learning | Reinforcement Learning |
| --- | --- | --- | --- |
| Discrete | Classification | Clustering | Simulation-based Optimization |
| Continuous | Regression | Reduction of Dimentionality | Autonomous Devices |


## What sort of method we can use or what is the right approach

| Problem | Approach | Why |
| --- | --- | --- |
| Detect whether a financial transaction is fraud. | Binary Classification | Only two possible outcomes: Fraud or Not Fraud |
| Predict the rate of deceleration of a car when brakes are applied. | Heuristic Approach (No ML Needed!, try not to stuff a problem that can be solved using heuristic approach into ML) | Well-known formulas involving speed, inertia and friction to predict this. |
| Determine the most efficient path of surface travel for a robotic lunar rover. | Simulation-based Reinforcement Learning | Must figure out the optimal path via trial, error and improvement. |
| Determine the breed of dog in a photograph. | Multi-Class Classification (Many breads and many options to choose) | Which dog breed is most associated with the picture among many breeds? |

## Cascading (stack) algorithms
- Sometimes it might require to stack algorithm on other to achieve the desired results.
- Example: Problem What is the estimated basket size of shoppers who responds to email promo:
  - First have to remove outliers (Random cut forest)
  - identify relevant attributes (features) that we are going to use from the data (PCA as learned earlier)
  - Then will have to group them into clusters (K-Means)
  - Apply an algorithm to predict the basket size (Linear Learner).  

## Confusion Matrix
- In case of a binary classification 
| | Actual Outcome TRUE | Actual Outcome FALSE |
| --- | --- | --- |
| Predicted outcome TRUE | I predicted correctly! | I was wrong. (False Positive) |
| Predicted outcome FALSE | I was wrong. (False Negative) | I predicted correctly! |

Lets take an example of outcome for this situation in case of bank

| | Actual Fraud | Actual Not Fraud |
|-|-|-|
| Predicted Outcome  Fraud | Happy Bank. Happy Customer. | Happy Bank. Angry Customer. |
| Predicted Outcome  Not Fraud | Angry Bank. Angry Customer. | Happy Bank. Happy Customer. |

Now lets check the financial situation with the bank 

|  | Actual Outcome Fraud | Actual Outcome Not Fraud |
| --- | --- | --- |
| Predicted outcome Fraud | No Money Loss | No Money Loss |
| Predicted outcome Not Fraud | Money Lost! | No Money Loss |

Evaluation approach:
What this means is that the bank is likely okay with False positives, but never okay with false negative.
In this case a false negative inferences are never okay. 

- Take another example of SPAM Filtering

| | Actual Outcome SPAM | Actual Outcome Not SPAM |
| --- | --- | --- |
| Predicted outcome SPAM | I predicted correctly! | I was wrong. (False Positive) |
| Predicted outcome Not SPAM | I was wrong. (False Negative) | I predicted correctly! |

| | Actual Outcome SPAM | Actual Outcome Not SPAM |
| --- | --- | --- |
| Predicted outcome SPAM | SPAM and it is blocked | Wrongly accusing email as spam and blocked legitimate email |
| Predicted outcome Not SPAM | SPAM are allowed with wrong prediction | Legitimate Email - goes through|

Evaluation approach:
In this case the legitimate emails are blocked with False positive. The end result will be okay to have the SPAM allowed (False Negatives)

## Data prep (how to send data to the model)
- How to make sure the model is correctly inferencing? 
  - A portion of the data is reserved for validation data called Testing Data (20%). 
  - Rest of the data called Training data is used from training the model (80%).
  - If the testing and training data differs too much there will be an increased error rates, to avoid such scenario before data splitting:
    - The data should be randomized to make sure it is equally mixed before splitting it into Training and testing data. This is to make sure both the testing and training data will have similar datasets.
  - Here is a sample code for splitting the data

```
import numpy as np
import os

# read raw data
print("Reading raw data from {}".format(raw_data_file))
raw = np.loadtxt(raw_data_file, delimiter=',')

# split into train/test with a 90/10 split
np.random.seed(0)
np.random.shuffle(raw)
train_size = int(0.9 * raw.shape[0])
train_features = raw[:train_size, :-1]
train_labels = raw[:train_size, -1]
test_features = raw[train_size:, :-1]
test_labels = raw[train_size:, -1]
```

  - If the data is time series data, like stock price over time (to understand google amzn stock from start until now)... How do we split such data into training and testing?
    - The time here has significant value, in this case we can slice of last few months for the training and testing data.
    - Keep in mind we cant use the same method for other types of data, doing so you might end up losing some important data set.
- **K FOLD**
- How do we know the splitting Training and testing data was best method to divide the data? 
- Assume the data is split into 4 (25%) each, the KFOLD will use any 3 portions to Train and last one to test. This process will be repeated and for all the models and keeps track of results. Then gives you the better suggestion on what is the right ML algorithm for that data to make inference.
This is an example of "4 FOLD".
- How many times that we are going to fold the data.
- Cross validation helps to test different machine learning algorithms.

## Training the model
- Sagemaker job can be submitted with Console, Jupyter, Sagemaker SDK, Apache Spark.
- In general Training Data + Testing Data will be copied at S3 and the model will read it from S3.

Assume you have below data and would like to predict the feature EVIL using a model 
As a first step how do you split the training and testing data ?

| Name | Affiliation | Evil |
| --- | --- | --- |
| Luke | Rebel | 0 |
| Leia | Rebel | 0 |
| Han | Rebel | 0 |
| Yoda | Rebel | 0 |
| Vadar | Empire | 1 |
| Jabba | Empire | 1 |
| Maul | Empire | 1 |
| JarJar | Empire | 1 |

How do we separate above data? starting at "Vadar" to below? Doing so will make all training data gets Label as 0, and testing with label as 1. 
How to fix that? We randomize it as below

| Name | Affiliation | Evil |
| --- | --- | --- |
| Luke | Rebel | 0 |
| JarJar | Empire | 1 |
| Leia | Rebel | 0 |
| Maul | Empire | 1 |
| Han | Rebel | 0 |
| Jabba | Empire | 1 |
| Yoda | Rebel | 0 |
| Vadar | Empire | 1 |

Also the Label needs to be moved to upfront 

| Evil | Name | Affiliation |
| --- | --- | --- |
| 0 | Luke | Rebel |
| 1 | JarJar | Empire |
| 0 | Leia | Rebel |
| 1 | Maul | Empire |
| 0 | Han | Rebel |
| 1 | Jabba | Empire |
| 0 | Yoda | Rebel |
| 1 | Vadar | Empire |

Once the data is ready, should upload to S3 so that a model can make use of it. 
- During upload the data format should be supported by the model's algorithm
- Most sagemaker accepts CSV - Content-Type = test/csv
- For unsupervised  - Content-Type = test/csv;label_size=0 (specify absence of a lable.)
- For optimal performance use protobuf-recordIO or pipemode. 

### How to train the model? 
- Use High-Level Python library provided by amazon sagemaker
- use Sagemaker SDK
1. Specify trainign algorithm 
2. Supply algorithm-specific hyper parameters
3. Specify input and output configuration

Hyper-parameters

| Hyperparameter | Values set before the learning process. Manage, adjust or tweak the learning process itself  |
| --- | --- |
| Parameter | Values derived via the learning process. |


## Sagemaker training

Why we need GPUs rather than CPUs for ML? 
> GPUs (Graphics Processing Units) are used in machine learning (ML) more than CPUs (Central Processing Units) primarily due to their parallel processing power. ML tasks, such as training deep neural networks, involve heavy matrix operations, which can be executed more efficiently on GPUs due to their architecture optimized for parallel computation. GPUs consist of thousands of cores compared to the limited number of cores in CPUs, enabling them to process multiple calculations simultaneously and handle large datasets much faster. As a result, using GPUs for ML tasks significantly reduces training time and improves performance compared to using CPUs alone.

The chips available?
Application Specific Integration Circuit - ASIC - (unable to change, Model burned into the chip, not flexible, costly)
Fix programed Gate Arrays(Specidic model programmed into the chip, hard to change, not flexible, costly)
GPU (Easy to change, flexible, affordable, optimized for general use)
CPU (Easy to change, flexible, cheap not optimized )


- A sample code to submit a model  on sagemaker

```python
from sagemaker import KMeans

data_location = 's3://{}/kmeans_highlevel_example/data'.format(bucket)
output_location = 's3://{}/kmeans_highlevel_example/output'.format(bucket)

kmeans = KMeans(role=role,
                train_instance_count=2,
                train_instance_type='ml.c4.8xlarge',
                output_path=output_location,
                k=10,
                data_location=data_location)

kmeans.fit(kmeans.record_set(train_set[0]))
```

| Code | Note |
| --- | --- |
| `from sagemaker import KMeans` | High-Level Python Library. We've chosen a KMeans Algorithm. |
| `data_location = 's3://{}/kmeans_highlevel_example/data'.format(bucket)` <br> `output_location = 's3://{}/kmeans_highlevel_example/output'.format(bucket)` | Define the location of training data. |
| `kmeans = KMeans(role=role,` <br> `                train_instance_count=2,` <br> `                train_instance_type='ml.c4.8xlarge',` <br> `                output_path=output_location,` <br> `                k=10,` <br> `                data_location=data_location)` | Instantiate the KMeans training object. <br> (Hyperparameter is k=10) |
| `kmeans.fit(kmeans.record_set(train_set[0]))` | Call the method to start the training job. |

## What happens once the job is submitted into the sagemaker?
- AWS has ECR repo with all the training and inference images.
  - Images with the tag ( :1) is used for Production which is the most stable version
  - Images with the tag ( :latest) is the most latest version
  - The images has detailks on what is the training input methods accepted (pipe/file) , file type and instance classes
- CreateTrainingJob call uses the training images - Uses more GPU intensive
- CreateModel API call ddeploys model into the inference image - Inference uses less resource intensive
- Once the code is submitted using the fit method ->
  - Sagemaker fetch the image and spin up the image, (Optionally you can use your custom Training image using the docker file)
  - It access the data from the S3
  - Finally when performs the deployment, it uses the inference image and deploys the trained model into the image.
- Cloudwatch will have the details of logs during the training.

## using Spark sagemaker library
- The data ingestion part will happen from Spark , where it creates the DataFrame and conert into Protobuf and stores into S3.
- Once the data is in S3 rest of the process are same as you normally do. 
- Once the model is deplyed, for the inference part the data will be referred from the Spark DataFrame.



# ML Algorithms
- Why do call algorithms "algorithms".
 - How to solve a class of problem by unambiguous (CLEAR) specification .
 - Set of steps to follow to solve a specific problem, intended to be repeatbale with same outcome.
 - Heuristic in otherhand is a educated guess, where the outcome is not guranteed and its baised (example does the dominos pizza taste good for me).
 - Computers can't make assumptions as they dont have such prior information like in Heuristic approach mentioned above. 
 - Algorithm is a finite process with known inputs to expected outputs by removing BIAS.
  - However BIAS can still creep into ML process
    - BIAS could be introducted from the data chosen to train the algorithm (Say you have removed a set of data unintentionally and used it for testing)
    - At the Feedback mechanism - Assume we set a feedback loop to ML but the feedback provided by us was based on Heuristic and most likely to see only one set of result. 

| Supervised Learning | Unsupervised Learning | Reinforcement Learning |
|----------------------|------------------------|------------------------|
| Training Data and Testing Data | No Training | How to Maximize Reward |
| Discrete | Classification | Clustering | Simulation-based Optimization |
| Continuous | Regression | Reduction of Dimentiionality | Autonomous Devices |

How the algorithms learns themselves?
## Supervised Learning**
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
### **Regression**
  Think about linear equation you learned at school y = 2x + 3. In this equation if value of either x is given the value of y can be easily figured out and viceversa. 
  We can also plot the value of x and y into a graph and should be able to predict the value by drawing a line.
#### **The Linear Regression** 
    - The linear models are supervised learning algorithms for regression, binary classification or multiclass classifcation problems. You give the models label y and x with x being high dimensional vectar and y us a numberic label. The algorithm learns a linear function or for classficvi ation problem a linear threshold function and maps a vector x to an approimation of label y. 
    In other words - (Collecting the data of height and weight (X/Y axis) and creating a more predictable linear line ). This is a traditional ML argorith,.. used on basic business cases.
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

 - Linear regression algorithm can be used to predict quantitative value based on given numeric input. 
  - Example Based on last five years of ROI from marketing spend, what can we expect to be this year's ROI
 - **Discrete Binary Classification** We can use Linear regression for classification problems, given that the data is structured in numeric format. 
  - Example based on past experience should I mail this customer or not. Predict yes or no 
 - **Descrete Multiclass Classififcation Problems** 
  - Example based on past customer response how should I reach out to customer, email, direct mail or phone call?
### Other Regression algorithms
- **Factorization Machines Algorithms**
  - Linear regression are good when there is a continuos data available. But if the data is missing we can use factorization machines. 
  - This is good for supervised learning algorithm for both binary classification and regression. Captures interaction between features with high dimensional sparse datasets. 
  - It is a good choice when you have holes in your data.
  - It analyses only relationship between two pairs of features at a time. 
  - CSV is not supported but file and pipe mode training are supported using recordIO-protobuf format with Float32 drivers
  - Doesn't support Multiclass problems. Cab run either on binary classification or regression mode.
  - Really needs lots of data, recommended dimesions space between 10,000 and 10,000,000
  - CPU is recommended with factorization 
  - Not good on dense data
  - use case?
    - Example click stream data on which ads on a webpage tend to be lcicked given known information about the person viewing the page. 
      - recommend a movie to a user based on other's rating. 
  - **How it works?**
    - Consider a moview recommendation engine
    - A user called Dante needs to get a movie recommended 
    - You have a dataset of different users rating movies called (clerks, Mallrats, Dogma, Clerks part 2)
    - But the data set from the users are incomplete because not all the users have saw the movie to get rated. 
    - The algithm can suggest the recommendation for the user.

### Classification**
- Predict a class label (category), discrete and unordered
- Can be binary (e.g. spam/not spam) or multi-class (e.g. letter recognition)
- Many classifiers can return a confidence per class
- The predictions of the model yield a decision boundary separating the classes
#### K-Nearest Neighbor is an eample algorithm 
- Predicts the value or classification based on which you are closest. It can be used to classify or to predict a value. 
- This is a supervised learning as the model has to be trained with a labelled data. 
- Example: 
  - Assume you have trained a classification model KNN to distinguish a shape (circular, triangle or square)
  - Now you are sending a new shape to classify. When you do so you set the hyperparameter K=2.
  - What this means is that, algorthm to find the two nearest neighbours to the input shape and decide to categrorize the input shape.
  - KNN you will pass the K value (neighbour) (how many objects are closest to the object that you are passing in )
  - KNN is a lazy algorithm - It doesn't use training data points to generalize but rather uses them to figure out who's nearby. 
  - KNN doesn't lean but rather uses the training dataset to decide on similar sample.
- Use cases:
  - Gruoup people together for credit risk based on attributes they share with others of known credit usage. 
  - Product recommendation on what someone likes, recommend similar items.

#### Image Analysis
- Another form of classification, it looks at the pixels of images. 
- Assume you have a data with two columns 
  - Column1: Image Data 
  - Column2: Label about the image. 
  - With the above data the model gets trained and upon providing a new image it performs the inference.
  - Usually image analysis tool will return a confidence.
- Amazon Rekognition. 
Common Image classification Algorithms:
##### **Image classification**. 
  - It decides the picture that you have given. It uses the convolutional neural network (ResNet) that can be trained from scrath or make use of transfer learning. 
  - Resources are used from imagenet - its a db online with labelled images. 
##### Object Detection
  - It will take a specific image and pull apart the objects inside the image. 
  - Example It can classify things on a desk
##### Semantic Segmentation
  - Low level analysis of individual pixels and identifies shapes within an image.
  - Example: Autonomous roboto going to a different plant, that can detect geographic structures, rocks hazards etc.
  - It accepets PNG as input
  - It supports GPU only for training..
  - It can deploy on CPU
    - Which means training can be done at Cloud and can be fit into any computers (like automobile, car, video camera etc)
  - Same like imagenet, the cityscapes dataset can be used to train and test self driving models to avoid people, cars etc. 
  - Image Metadata Extraction:
    - The system can be used for extracting a scene from the image and store it in a file so the image could be searchable. 
  - Computer vision systems: 
    - Recognize orientation of a part on an assembly line and issue a command gto robotic arn to re-orient the part.

##### Blazing Text 
- It is based on fast text algo developed by Facebook. 
- Inference can be done in real time, optimized way to determine contextual sementic relationships between words in a body of text. 

| Modes | Word2Vec (Unsupervised) | Text Classification (Supervised) |
| --- | --- | --- |
| Single CPU Instance | Continuous Bag of Words <br> Skip-gram <br> Batch Skip-gram | Supervised |
| Single GPU Instance (1 or more GPUs) | Continuous Bag of Words <br> Skip-gram | Supervised with 1 GPU |
| Multiple CPU Instances | Batch Skip-gram | None |

##### Object2Vec
- A way to map out things in a d-dimentional space to figure out how similar they might be to one another.
- Example, give a set of words to understand how close they are . Example, Sad, upset, angry lonely, scared, happy, appreciative, fun
- It can also help to group things together based on interest. 
- It expects things in pairs.
- Enbedding is used in pre-processing to pass the data.
- Training data is required as this is a supervised algorithm.
- It can predict rating given by a person on a movie based on similar ratings he given on other movies. 
- It can determine which genre a book is based on its similarity to known genre.



## Unsupervised Learning**
  - Looking for patterns in the data, in which we dont necessarly see a pattern. 
  - Imagine we have a data set with scores in x and y axis and there is no relation for the data.
    - - Find a relation ship on data that organic brain necesarily dont know the relation. 
  - It is best used when many dimensions are available in the data. 
  - Unlabeled data, or data with unknown structure
  - Explore the structure of the data to extract information
  - Many types, we’ll just discuss two.
#### **Clustering** 
    - Organize information into meaningful subgroups (clusters)
    - Objects in cluster share certain degree of similarity (and dissimilarity to other clusters)
    - Example: distinguish different types of customers.
    - Example: Assume you have a cluster of shapes, circle, square, and triangle, the algorithm can Group object based on the number of sides. 
    - K-Means is an example for Clustering Algorithm
      - K-Means expects tabular data. It is unsupervised
      - You need to supply the column of the data which explains the attributes and which attribute the algorithm needs to pay attention on
      - You should know data well to propose the attributes to the algorithm, if you have no idea there are ways around the tools.
      - Sagemaker users modified K - Means
      - Sagemaker recommends to use CPU instances and can only use 1 GPU 
      - Spend some time to make sure you have choosen the right attribute is chosen to make the prediction.
    - K-Means use case:
      - Audio wave form convert to digital 
      - MNIST - predict the handwritten image to digits
#### **Dimensionality reduction**
    - Data can be very high-dimensional and difficult to understand, learn from, store,…
    - Dimensionality reduction can compress the data into fewer dimensions, while retaining most of the information
    - Contrary to feature selection, the new features lose their (original) meaning
    - The new representation can be a lot easier to model (and visualize)
    - Used in unsupervised machine learning type.
#### Anomaly Detection
  - What is not the same, what is the different, what is not usual 
##### Random Cut Forest Algorithm
  - Human eyes are set to recognize if something changes in a picture (two picture similar but a small change (a person missing)). 
  - Computers dont have the same perception ability like human but have to synthesis with mathematics
  - Randomcut forest algorithm is useful in understanding occurences in the data that are significantly beyond normal. (Usually more than 3 standard deviations). 
  - Note sometimes this outlier can mess up the training data.
  - Random Cut forest algorithm gives an anomaly score to data points, Low scores indicate that a data point is considered **Normal** while high score indicates the presence of an anomaly.
  - It scales well - RCF scales very well wiuth respct to number of features dataset size and number of instances.
  - Doesn't benifit from GPU and AWS recommends to use compute instances
  - Example:
   - Lets assime you have scatter plot and a single outlier at extreme corner,
   - When the algorithm makes the cut the one side will have the actual data and other side will have the outlier. 
  - Use Case: 
    - Fraud detection, check transaction happening at unusual place, at unusal time flag the transaction for closer look. 
    - Quality control, Analyze an audio test pattern played by a high-end speaker system for any unusual frequency.
###### IP Insights,
    - It can use the historic base line for the IP pattern by users IDs and account numbers.  Then flag odd online behavior.
    - Use cases: If a user tries to loginto a website from an anomalous IP address it can trigger an additional two factor authentication. 
    - Fraud detection: ON banking only allow user activites from a known UP range.

#### Text analysis Algorithms.
##### Latent Dirichlet Allocation (LDA)
- How similar two documents are based on the similar words in the documents. 
-  Example: Article Recommendation:
  - Recommend articles on similar topics which you might ave read or rated in the past. 
- Example: Musical Influence modelling:
  - Identify the most influencial artist in the given time. 
##### Nueral Topic Model.
- Similar to LDA in that both NTM and LDA can perform topic modeling. However NTM uses a different algo which might yield different results that LDA.
##### Seuence to Sequence (Seq2Seq)
- Think a language translation engine that can take in some text and predict what that text might be in another language. We must supply training data and vocabulary. Most of the language translation engines
- Speech to text conversion is another use case.

##### Blazing Text 
- It is based on fast text algo developed by Facebook. 
- Inference can be done in real time, optimized way to determine contextual sementic relationships between words in a body of text. 
- Can be used for Sentimental analysis
- Can be used to classify the documentaiton.
- AWS uses same for Amazon Comprehend - for sentiment analysis, Amazon Macie for analyzing sensitive information.

| Modes | Word2Vec (Unsupervised) | Text Classification (Supervised) |
| --- | --- | --- |
| Single CPU Instance | Continuous Bag of Words <br> Skip-gram <br> Batch Skip-gram | Supervised |
| Single GPU Instance (1 or more GPUs) | Continuous Bag of Words <br> Skip-gram | Supervised with 1 GPU |
| Multiple CPU Instances | Batch Skip-gram | None |


## Semi Supervised Learning**
  - learn a model from (few) labeled and (many) unlabeled examples

## **Reinforcement Learning**
  - It is used in robotics and automation. 
  - Try to maximize the reward
  - For example, use reward machanism, to train the model. For example, a robot is asked to pickup pengiun and if it picks wrong one we give a negative reward. The model try to get more reward. 
  - AWS Deepracer uses reinforcement learning. 
  - Develop an agent that improves its performance based on interactions with the environment
  - Example: games like Chess, Go,…
  - Search a (large) space of actions and states
  - Reward function defines how well a (series of) actions works
  - Learn a series of actions (policy) that maximizes reward through exploration
### Markov Decision Process MDP
- Agent to place in an environment assume its 6x7 grid. 
- The agent to get to a candy at location G6 which is the goal
- Note, most of the Reinforcement learning will not have a goal but the aim is to accumulate the most rewards.
- State: The information of the environment. Lets assume agent is in C3 position in the grid.
- Action: is movement of the agent - up/down to move to the destination
- Observation: Information available on agent on each step. 
- Episodes: iteration from start to end while it accumulate the award. 
- After a series of error and trail it will develop the optimized path to reach to the destination this is called the **Policy**.
- Example AWS Deeplearning code

```
def reward_function(self, on_track, x, y, distance_from_center, car_orientation, progress, steps, throttle, steering, track_width, waypoints, closest_waypoints):

    reward = 1e-3
    marker_1 = 0.1 * track_width
    marker_2 = 0.2 * track_width
    marker_3 = 0.4 * track_width

    if distance_from_center >= 0.0 and distance_from_center <= marker_1:
        reward = 1
    elif distance_from_center >= marker_1 and distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center >= marker_2 and distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3 # likely crashed/ close to off track

    return reward
```
- This can be used with AWS Robomaker
- Keep an eye to make sure that the Policy is continue to evolve.
- Use case - Autonomous vehicle. 
  - Control the airconditioning in a room
## Forecasting
- Past performance is not really an indicator of future results. We cant rely on past performance to predict the future.
- To mitigate that sagemaker has following algorithm 
### DeepAR
- It is optimized to work with complex forecasting problems. 
- It can predict both point-in-time values and estimated values over a timeframe by using multiple sets of historic data.
[refer more data]
- Use case: Forecast new product performance
- Preduct Labor needs for special events

## Ensemble learning
- Using Muiltiple learning algorithms and models collectively to hopefully improve the model acuracy
### Extreme Gradient Boosting (XGBoost)
- XGBoost is a supervised learning. 
- It accepts CSV and Libsvm for trainign inference. 
- Trains only on CPU and memory bound. 
- Needs lot of memory 
- Use case: What price should be ask for a house that we need to sell. Consider all the variables comeinto picture. 
  - Use Decision Tree Ensembles to calculate through different layer approach to create a model to predict the 
  - Ranking products On a e-commerce sort the search results based on user data.
  - 



## **The Logistic Regression**(Created for Yes or No scenarios), a binary output.
## **Support Vector machine**. (Cluster information into categories)
## **Decision Trees** (Decisioning flow) 


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

































#----------------------#



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
