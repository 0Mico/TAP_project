FROM docker.elastic.co/elasticsearch/elasticsearch:8.15.0

USER root

COPY create_index.sh /usr/local/bin/create_index.sh

RUN chown root:root /usr/local/bin/create_index.sh
RUN chmod +x /usr/local/bin/create_index.sh

CMD ["sh", "-c", "/usr/local/bin/create_index.sh & elasticsearch"]