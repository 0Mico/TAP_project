import os
import torch
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf
from pyspark.sql.types import (
    StructType, StructField, StringType, ArrayType
)
from transformers import AutoTokenizer, AutoModelForTokenClassification


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
            return []
        
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
            
            # Predict
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
            
            return list(skills)
        
        except Exception as e:
            print(f"Error extracting skills: {e}")
            return []


def create_spark_session():
    """Create and configure Spark session"""

    spark = SparkSession \
        .builder \
        .appName("JobPostSkillsExtractor") \
        .config("spark.sql.streaming.checkpointLocation", "/tmp/spark-checkpoint") \
        .getOrCreate()
    
    # Reduce logging verbosity
    spark.sparkContext.setLogLevel("WARN")
    
    return spark


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


def main():
    print("="*60)
    print("JOB POST SKILLS EXTRACTOR - SPARK STREAMING")
    print("="*60)
    
    # Get configuration from environment variables
    kafka_bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    elasticsearch_host = os.environ.get("ELASTICSEARCH_HOST", "elasticsearch")
    elasticsearch_port = os.environ.get("ELASTICSEARCH_PORT", "9200")
    
    print(f"\nConfiguration:")
    print(f"  Kafka Bootstrap Servers: {kafka_bootstrap_servers}")
    print(f"  Elasticsearch: {elasticsearch_host}:{elasticsearch_port}")
    print(f"  Kafka Topic: deduped_job_posts")
    print(f"  Elasticsearch Index: job_posts_with_skills")
    print("="*60 + "\n")
    
    # Create Spark session
    spark = create_spark_session()
    
    # Register UDF for skill extraction
    extract_skills_udf = udf(SkillsExtractor.extract_skills, ArrayType(StringType()))
    
    # Read from Kafka as a streaming DataFrame
    print("Connecting to Kafka...")
    kafka_df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
        .option("subscribe", "deduped_job_posts") \
        .option("startingOffsets", "earliest") \
        .option("failOnDataLoss", "false") \
        .load()
    
    print("✓ Connected to Kafka\n")
    
    # Parse JSON from Kafka messages
    schema = get_kafka_schema()
    
    print("Processing job posts...")
    parsed_df = kafka_df \
        .select(from_json(col("value").cast("string"), schema).alias("data")) \
        .select("data.*")
    
    # Extract skills from description, then drop the description
    enriched_df = parsed_df \
        .withColumn("Skills", extract_skills_udf(col("Description"))) \
        .drop("Description")
    
    # Write to Elasticsearch
    query = enriched_df \
        .writeStream \
        .outputMode("append") \
        .format("org.elasticsearch.spark.sql") \
        .option("es.nodes", elasticsearch_host) \
        .option("es.port", elasticsearch_port) \
        .option("es.resource", "job_posts_with_skills") \
        .option("es.mapping.id", "Job_ID") \
        .option("es.write.operation", "upsert") \
        .option("es.nodes.wan.only", "true") \
        .option("checkpointLocation", "/tmp/spark-checkpoint") \
        .start()
    
    print("✓ Streaming started successfully!")
    print("\nWaiting for job posts from Kafka...")
    print("Press Ctrl+C to stop\n")
    
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