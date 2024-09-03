function is_elasticsearch_up {
  curl -s "localhost:9200" | grep "You Know, for Search" > /dev/null
  return $?
}

while ! is_elasticsearch_up; do
  sleep 2
done

# Create a new index
curl -X PUT "localhost:9200/deduped_posts"