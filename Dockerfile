FROM alpine:latest

USER root

# Install curl
RUN apk add --no-cache curl

# Copy the script into the container
COPY ./Elasticsearch/create_index.sh /usr/local/bin/create_index.sh

# Make sure the script is executable
RUN chmod +x /usr/local/bin/create_index.sh

# Entry point or command to execute the script
ENTRYPOINT ["/usr/local/bin/create_index.sh"]