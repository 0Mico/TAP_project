FROM apache/spark-py:v3.4.0

USER root

RUN apt-get update && \
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pyspark==3.4.0 \
    elasticsearch==8.15.0 \
    torch \
    transformers

WORKDIR /app

COPY Spark-app.py /app/

# Run the Spark application
CMD ["/opt/spark/bin/spark-submit", \
     "--master", "spark://spark-master:7077", \
     "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.elasticsearch:elasticsearch-spark-30_2.12:8.15.0", \
     "--conf", "spark.driver.memory=1g", \
     "--conf", "spark.executor.memory=2g", \
     "/app/Spark-app.py"]