# Skills extraction pipeline

This project consists in an event-driven pipeline to extract hard skills required by companies in linkedin job posts.

It processes data in real time and according to new events (job posts) refresh a kibana dashboards that shows insights about skills and their relationship with professions, industry types, etc...

It is fully managed using docker-compose

## Architecture

![Infrastructure Diagram](images/infrastructure.png)

The app contains a scraper used as a data source.

The scraper uses Logstash to ingest data in:

- an Elasticsearch index to remove duplicates and to make sure that the posts that go ahead in the pipeline contain the description field, from where the skills are extracted;

- a Kafka topic from where the posts will be read by Spark, that handles the normalization and categorization of some fields of the job posts, and then extracts the skills.

Spark then writes the enriched data to an Elsticsearch index (different from the one used by Logstash).

At the end of the pipeline is used Kibana to show insights about the processed posts.

## Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM available for containers
- Sufficient disk space for Elasticsearch data and Kafka logs

## Getting Started

1. **Clone the repository**
```bash
git clone <repository-url>
cd <TAP_project>
```

2. **Prepare the skill extraction model**
   - Place your trained BERT model in `./Spark/Model/skill_extractor_model/`
   - The model should be compatible with HuggingFace Transformers

3. **Configure normalization and categorization**
   - Update `job_titles_normalization_map.json` for job titles mapping
   - Update `skills_normalization_map.json` for skills name mapping
   - Update `cloud_services.json` and `technologies.json` for skills categorization

4. **Start the pipeline**
```bash
docker-compose up -d
```
5. **Monitor the services**
   - Kafka UI: http://localhost:8080
   - Kibana: http://localhost:5601
   - Spark Master UI: http://localhost:8081

## Troubleshooting

- **Scraper fails with 429 errors**: LinkedIn is rate limiting. The scraper has exponential backoff, but you may need to reduce scraping frequency
- **Spark fails to load model**: Ensure the model is properly mounted in `/app/model`
- **Elasticsearch health check fails**: Increase memory allocation or wait longer for initialization
- **Checkpoint errors in Spark**: The checkpoint directory is cleared on startup to force reprocessing