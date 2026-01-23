import os
import torch
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf, lower as spark_lower
from pyspark.sql.types import (StructType, StructField, StringType, ArrayType)
from transformers import AutoTokenizer, AutoModelForTokenClassification
import shutil

from SkillsNormalizer import SkillNormalizer
from SkillsCategorizer import SkillCategorizer
from JobTitlesNormalizer import JobTitleNormalizer

class SkillsExtractor:
    """Class to load the model only once per executor for efficiency"""

    _model = None
    _tokenizer = None
    _id2label = None
    
    @classmethod
    def get_model(cls):
        """Load model once per executor and reuse it"""

        if cls._model is None:
            model_path = "/app/model"
            print(f"Loading model from {model_path}...")
            
            cls._tokenizer = AutoTokenizer.from_pretrained(model_path)
            cls._model = AutoModelForTokenClassification.from_pretrained(model_path)
            cls._model.eval()  # Set to evaluation mode
            cls._id2label = cls._model.config.id2label
            
            print(f"✓ Model loaded successfully!")
            print(f"Labels: {list(cls._id2label.values())}")
        
        return cls._model, cls._tokenizer, cls._id2label
    

    @classmethod
    def extract_skills(cls, text):
        """ Extract skills from job description text using BIO tagging """

        if not text or not text.strip():
            print('RECEIVED EMPTY TEXT')
            return {
                'raw_skills': [],
                'categories': SkillCategorizer._get_empty_categories(),
                'cloud_services': SkillCategorizer._get_empty_cloud_services()
            }
        
        try:
            model, tokenizer, id2label = cls.get_model()
            
            # Tokenize - use max_length to handle long descriptions
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=False
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
            
            # Get tokens and labels
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            predicted_labels = [id2label[p.item()] for p in predictions[0]]
            
            # Extract skills using BIO tagging
            skills = set()
            current_skill = []
            
            for token, label in zip(tokens, predicted_labels):
                if token in ['[CLS]', '[SEP]', '[PAD]']:
                    continue
                
                if label == "B-SKILL":
                    # Save previous skill if exists
                    if current_skill:
                        skill_text = tokenizer.convert_tokens_to_string(current_skill).strip()
                        if skill_text:
                            skills.add(skill_text)
                    # Start new skill
                    current_skill = [token]
                    
                elif label == "I-SKILL" and current_skill:
                    # Continue current skill
                    current_skill.append(token)
                    
                else:
                    # End of skill (O tag)
                    if current_skill:
                        skill_text = tokenizer.convert_tokens_to_string(current_skill).strip()
                        if skill_text:
                            skills.add(skill_text)
                        current_skill = []
            
            # Last skill
            if current_skill:
                skill_text = tokenizer.convert_tokens_to_string(current_skill).strip()
                if skill_text:
                    skills.add(skill_text)
            
            raw_skills = list(skills)
            normalized_skills = SkillNormalizer.normalize_skills_list(raw_skills)
            final_skills = SkillCategorizer.categorize_skills(normalized_skills)
            
            return final_skills
        
        except Exception as e:
            print(f"Error extrcting skills: {e}")            
            return {
                'raw_skills': [],
                'categories': SkillCategorizer._get_empty_categories(),
                'cloud_services': SkillCategorizer._get_empty_cloud_services()
            }
 
    

def create_spark_session():
    spark = SparkSession \
        .builder \
        .appName("JobPostSkillsExtractor") \
        .config("spark.sql.streaming.checkpointLocation", "/tmp/spark-checkpoint") \
        .getOrCreate() 
    return spark


def connect_to_kafka(spark, kafka_bootstrap_servers):
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
        .option("subscribe", "deduped_job_posts") \
        .option("startingOffsets", "earliest") \
        .option("failOnDataLoss", "false") \
        .load()
    return kafka_df


def get_kafka_schema():
    """ Define schema for incoming Kafka messages"""
    return StructType([
        StructField("Job_ID", StringType(), False),
        StructField("Title", StringType(), False),
        StructField("Company_name", StringType(), False),
        StructField("Location", StringType(), False),
        StructField("Pubblication_date", StringType(), False),
        StructField("Description", StringType(), False),
        StructField("Seniority_level", StringType(), False),
        StructField("Employment_type", StringType(), False),
        StructField("Job_Function", StringType(), False),
        StructField("Industry_type", StringType(), False)
    ])

def get_skills_schema():
    """ Defines the Spark StructType that matches the dictionary returned by SkillCategorizer """
    return StructType([
        StructField("raw_skills", ArrayType(StringType()), True),
        
        StructField("categories", StructType([
            StructField("cloud_providers", ArrayType(StringType()), True),
            StructField("programming_languages", ArrayType(StringType()), True),
            StructField("frontend_frameworks", ArrayType(StringType()), True),
            StructField("backend_frameworks", ArrayType(StringType()), True),
            StructField("databases", ArrayType(StringType()), True),
            StructField("devops_tools", ArrayType(StringType()), True),
            StructField("monitoring_tools", ArrayType(StringType()), True),
            StructField("testing_frameworks", ArrayType(StringType()), True)
        ]), True),
        
        StructField("cloud_services", StructType([
            StructField("aws", ArrayType(StringType()), True),
            StructField("gcp", ArrayType(StringType()), True),
            StructField("azure", ArrayType(StringType()), True),
            StructField("ibm", ArrayType(StringType()), True),
            StructField("oracle", ArrayType(StringType()), True)
        ]), True)
    ])

def write_to_elasticsearch(enriched_df, host, port):
    query = enriched_df.writeStream \
        .outputMode("append") \
        .format("org.elasticsearch.spark.sql") \
        .option("es.nodes", host) \
        .option("es.port", port) \
        .option("es.resource", "job_posts_with_skills") \
        .option("es.mapping.id", "Job_ID") \
        .option("es.write.operation", "upsert") \
        .option("es.nodes.wan.only", "true") \
        .option("checkpointLocation", "/tmp/spark-checkpoint") \
        .start()
    return query



def main():  
    if os.path.exists("/tmp/spark-checkpoint"):
        print("⚠️ DELETING CHECKPOINT at /tmp/spark-checkpoint to force re-processing...")
        shutil.rmtree("/tmp/spark-checkpoint")
    
    kafka_bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    elasticsearch_host = os.environ.get("ELASTICSEARCH_HOST", "elasticsearch")
    elasticsearch_port = os.environ.get("ELASTICSEARCH_PORT", "9200")
    
    print(f"\nConfiguration:")
    print(f"  Kafka Bootstrap Servers: {kafka_bootstrap_servers}")
    print(f"  Elasticsearch: {elasticsearch_host}:{elasticsearch_port}")
    print(f"  Kafka Topic: deduped_job_posts")
    print(f"  Elasticsearch Index: job_posts_with_skills")
    print("="*60 + "\n")
    
    spark = create_spark_session()
    
    # Load data schemas
    kafka_messages_schema = get_kafka_schema()
    final_skills_format = get_skills_schema()
    
    # Register UDF functions
    extract_skills_udf = udf(SkillsExtractor.extract_skills, final_skills_format)
    title_normalization_udf = udf(JobTitleNormalizer.normalize_job_title, StringType())
    
    # Read from Kafka
    print("Connecting to Kafka...")
    kafka_df = connect_to_kafka(spark, kafka_bootstrap_servers)
    print("✓ Connected to Kafka\n")
    
    print("Processing job posts...")
    parsed_df = kafka_df \
        .select(from_json(col("value").cast("string"), kafka_messages_schema).alias("data")) \
        .select("data.*")
    
    # Make all fields lowercase
    parsed_df = parsed_df \
        .withColumn("Title", spark_lower(col("Title"))) \
        .withColumn("Company_name", spark_lower(col("Company_name"))) \
        .withColumn("Location", spark_lower(col("Location"))) \
        .withColumn("Seniority_level", spark_lower(col("Seniority_level"))) \
        .withColumn("Employment_type", spark_lower(col("Employment_type"))) \
        .withColumn("Job_Function", spark_lower(col("Job_Function"))) \
        .withColumn("Industry_type", spark_lower(col("Industry_type")))
    
    # Normalize job titles
    parsed_df = parsed_df \
        .withColumn("Title", title_normalization_udf(col("Title")))
    
    # Extract skills from description, then drop the description
    enriched_df = parsed_df \
        .withColumn("Skills", extract_skills_udf(col("Description"))) \
        .drop("Description")
    
    query = write_to_elasticsearch(enriched_df, elasticsearch_host, elasticsearch_port)
    
    # Wait for termination
    query.awaitTermination()



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        raise