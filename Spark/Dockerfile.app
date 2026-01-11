FROM apache/spark-py:v3.4.0

USER root

RUN apt-get update && \
    apt-get install -y python3-pip zip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pyspark==3.4.0 \
    elasticsearch==9.1.0 \
    torch \
    transformers

WORKDIR /app

COPY config/ /app/config/
COPY SkillsNormalizer.py /app/
COPY SkillsCategorizer.py /app/
COPY Spark-app.py /app/

# Run the Spark application
CMD ["/opt/spark/bin/spark-submit", \
     "--master", "spark://spark-master:7077", \
     "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.elasticsearch:elasticsearch-spark-30_2.12:9.1.0", \
     "--files", "/app/config/normalization_map.json,/app/config/cloud_services.json,/app/config/technologies.json", \
     "--py-files", "/app/SkillsNormalizer.py,/app/SkillsCategorizer.py", \
     "--conf", "spark.driver.memory=1g", \
     "--conf", "spark.executor.memory=2g", \
     "/app/Spark-app.py"]