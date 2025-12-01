FROM bitnami/spark:3.5.0

USER root

RUN pip install --no-cache-dir \
    pyspark==3.5.0 \
    elasticsearch==8.15.0 \
    torch \
    transformers

WORKDIR /app

COPY Spark-app.py /app/

# The model will be mounted as a volume at runtime

USER 1001

# Run the Spark application
CMD ["spark-submit", \
     "--master", "spark://spark-master:7077", \
     "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.elasticsearch:elasticsearch-spark-30_2.12:8.15.0", \
     "--conf", "spark.driver.memory=1g", \
     "--conf", "spark.executor.memory=2g", \
     "/app/Spark-app.py"]