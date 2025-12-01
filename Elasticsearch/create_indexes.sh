#!/bin/sh

create_index_if_not_exists() {
  INDEX_NAME=$1
  MAPPING=$2
  
  if curl -s -o /dev/null -w "%{http_code}" "http://elasticsearch:9200/$INDEX_NAME" | grep -q "404"; then
    echo "Creating index: $INDEX_NAME"
    curl -X PUT "http://elasticsearch:9200/$INDEX_NAME" \
      -H 'Content-Type: application/json' \
      -d "$MAPPING"
  else
    echo "Index $INDEX_NAME already exists, skipping..."
  fi
}

# Create deduped_posts index (raw job posts with full details)
create_index_if_not_exists "deduped_posts" '{
  "mappings": {
    "properties": {
      "Job_ID": {"type": "keyword"},
      "Title": {"type": "text"},
      "Company_name": {"type": "keyword"},
      "Location": {"type": "keyword"},
      "Pubblication_date": {"type": "date", "format": "yyyy-MM-dd"},
      "Description": {"type": "text"},
      "Seniority_level": {"type": "keyword"},
      "Employment_type": {"type": "keyword"},
      "Job_Function": {"type": "keyword"},
      "Industry_type": {"type": "keyword"}
    }
  }
}'

# Create job_posts_with_skills index (processed posts with extracted skills)
create_index_if_not_exists "job_posts_with_skills" '{
  "mappings": {
    "properties": {
      "Job_ID": {"type": "keyword"},
      "Title": {"type": "text"},
      "Company_name": {"type": "keyword"},
      "Location": {"type": "keyword"},
      "Pubblication_date": {"type": "date", "format": "yyyy-MM-dd"},
      "Skills": {"type": "keyword"},
      "Seniority_level": {"type": "keyword"},
      "Employment_type": {"type": "keyword"},
      "Job_Function": {"type": "keyword"},
      "Industry_type": {"type": "keyword"}
    }
  }
}'

echo "All indices ready!"