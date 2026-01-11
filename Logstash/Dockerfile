FROM logstash:9.1.9

RUN logstash-plugin install logstash-filter-elasticsearch \
    && logstash-plugin install logstash-output-elasticsearch

COPY logstash.yml /usr/share/logstash/config/
COPY deduplication.conf /usr/share/logstash/pipeline/

EXPOSE 5000

CMD ["logstash", "-f", "/usr/share/logstash/pipeline/deduplication.conf"]
