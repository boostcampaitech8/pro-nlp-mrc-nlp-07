#!/bin/bash

# 1. Version Setting (Using stable 8.11.1)
ES_VERSION="9.2.1"
ES_DIR="elasticsearch-${ES_VERSION}"
DOWNLOAD_URL="https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-${ES_VERSION}-linux-x86_64.tar.gz"

echo "🔹 Elasticsearch ${ES_VERSION} Setup Starting..."

# 0. Cleanup: Stop existing instance
if [ -f "$ES_DIR/es_pid" ]; then
    PID=$(cat "$ES_DIR/es_pid")
    if ps -p $PID > /dev/null; then
        echo "🛑 Stopping currently running Elasticsearch (PID: $PID)..."
        kill $PID
        while ps -p $PID > /dev/null; do
             sleep 1
             echo -n "."
        done
        echo " Stopped."
    fi
    rm "$ES_DIR/es_pid"
fi

# 2. Download
if [ ! -d "$ES_DIR" ]; then
    if [ ! -f "elasticsearch-${ES_VERSION}-linux-x86_64.tar.gz" ]; then
        echo "Downloading Elasticsearch..."
        wget -q "$DOWNLOAD_URL"
        if [ $? -ne 0 ]; then
            echo "❌ Download failed! Check internet connection."
            exit 1
        fi
    fi
    echo "Extracting..."
    tar -xzf "elasticsearch-${ES_VERSION}-linux-x86_64.tar.gz"
    rm "elasticsearch-${ES_VERSION}-linux-x86_64.tar.gz"
fi

# 2.5 Install Nori Plugin (Korean Analyzer)
if [ -d "$ES_DIR" ]; then
    echo "Installing analysis-nori plugin..."
    # Check if already installed
    if "$PWD/$ES_DIR/bin/elasticsearch-plugin" list | grep -q "analysis-nori"; then
        echo "✅ analysis-nori is already installed."
    else
        "$PWD/$ES_DIR/bin/elasticsearch-plugin" install analysis-nori --batch
        echo "✅ analysis-nori installed."
    fi
fi

# 3. Configure (Disable Security for Dev)
CONFIG_FILE="$ES_DIR/config/elasticsearch.yml"
echo "Configuring elasticsearch.yml..."

cat <<EOF > "$CONFIG_FILE"
xpack.security.enabled: false
xpack.security.enrollment.enabled: false
xpack.security.http.ssl.enabled: false
xpack.security.transport.ssl.enabled: false
http.host: 0.0.0.0
discovery.type: single-node
EOF

# 4. Start
echo "🚀 Starting Elasticsearch (Daemon mode)..."
su - es-user -c "$PWD/$ES_DIR/bin/elasticsearch -d -p $PWD/$ES_DIR/es_pid"

# 7. 실행 확인
echo "Waiting for Elasticsearch to boot..."
for i in {1..60}; do
    if curl -s "http://localhost:9200" > /dev/null; then
        echo "✅ Elasticsearch is running at http://localhost:9200"
        exit 0
    fi
    sleep 1
    echo -n "."
done

echo ""
echo "❌ Failed to start. Check logs below:"
cat "$ES_DIR/logs/elasticsearch.log"