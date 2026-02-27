#!/bin/bash
# Oracle Cloud VM setup script for Mancini trading bot
# Run as root on a fresh Ubuntu 22.04 ARM instance
set -e

echo "=== Mancini Trading Bot — Cloud Setup ==="

# Install Docker
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    apt-get update
    apt-get install -y docker.io docker-compose
    systemctl enable docker
    systemctl start docker
    echo "Docker installed."
else
    echo "Docker already installed."
fi

# Clone or update repo
REPO_DIR="/opt/mancini"
if [ -d "$REPO_DIR" ]; then
    echo "Updating existing repo..."
    cd "$REPO_DIR" && git pull
else
    echo "Cloning repo..."
    echo "NOTE: You need to provide your git repo URL."
    echo "  git clone <your-repo-url> $REPO_DIR"
    echo "  OR: scp -r . root@<vm-ip>:$REPO_DIR"
    mkdir -p "$REPO_DIR"
fi

# Setup .env
if [ ! -f "$REPO_DIR/.env" ]; then
    cp "$REPO_DIR/deploy/.env.example" "$REPO_DIR/.env"
    echo ""
    echo "IMPORTANT: Edit $REPO_DIR/.env with your IB credentials:"
    echo "  nano $REPO_DIR/.env"
    echo ""
fi

# Create logs directory
mkdir -p "$REPO_DIR/logs"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit credentials:  nano $REPO_DIR/.env"
echo "  2. Start the bot:     cd $REPO_DIR && docker-compose up -d"
echo "  3. Check logs:        docker-compose logs -f mancini-bot"
echo "  4. Check IB Gateway:  docker-compose logs -f ib-gateway"
echo "  5. Stop everything:   docker-compose down"
echo ""
