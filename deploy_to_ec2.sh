#!/bin/bash

# EC2 Docker Deployment Script with Architecture Fix
# This script deploys your Docker container to AWS EC2

set -e  # Exit on error

echo "🚀 Starting EC2 Deployment"
echo "=========================="

# Configuration
EC2_HOST="ec2-34-201-166-105.compute-1.amazonaws.com"
EC2_USER="ubuntu"
KEY_FILE="$HOME/Desktop/Fraud_Detection_System/AI_Training_VM_key.pem"
DOCKER_IMAGE_NAME="fraud-detection-api"
DOCKER_IMAGE_TAG="latest"
CONTAINER_NAME="fraud-api"
CONTAINER_PORT=8000
HOST_PORT=80

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Step 1: Verify key file exists
echo ""
echo "Step 1: Verifying SSH key..."
if [ ! -f "$KEY_FILE" ]; then
    print_error "Key file not found at $KEY_FILE"
    echo "Please ensure AI_Training_VM_key.pem is at $KEY_FILE"
    exit 1
fi

# Set correct permissions
chmod 400 "$KEY_FILE"
print_info "Key file found and permissions set"

# Step 2: Test SSH connection
echo ""
echo "Step 2: Testing SSH connection..."
if ssh -i "$KEY_FILE" -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" "echo 'Connection successful'" &> /dev/null; then
    print_info "SSH connection successful"
else
    print_error "Cannot connect to EC2 instance"
    echo "Please check:"
    echo "  1. Your internet connection"
    echo "  2. EC2 instance is running"
    echo "  3. Security group allows SSH (port 22)"
    exit 1
fi

# Step 3: Build Docker image for AMD64 architecture
echo ""
echo "Step 3: Building Docker image for AMD64 architecture..."
print_warning "Building for linux/amd64 platform (EC2 compatible)"

# Check if running on Apple Silicon
if [[ $(uname -m) == "arm64" ]]; then
    print_info "Detected ARM64 (Apple Silicon) - building cross-platform image"
fi

# Build for AMD64
docker build --platform linux/amd64 -t "$DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG" .

# Verify architecture
ARCH=$(docker image inspect "$DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG" | grep -o '"Architecture": "[^"]*"' | cut -d'"' -f4)
print_info "Image built for architecture: $ARCH"

if [ "$ARCH" != "amd64" ]; then
    print_error "Image architecture is $ARCH, expected amd64"
    exit 1
fi

# Step 4: Save Docker image
echo ""
echo "Step 4: Saving Docker image..."
docker save "$DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG" | gzip > /tmp/fraud-detection.tar.gz
IMAGE_SIZE=$(du -h /tmp/fraud-detection.tar.gz | cut -f1)
print_info "Docker image saved (Size: $IMAGE_SIZE)"

# Step 5: Transfer image to EC2
echo ""
echo "Step 5: Transferring Docker image to EC2..."
echo "This may take a few minutes depending on image size..."
scp -i "$KEY_FILE" -o StrictHostKeyChecking=no /tmp/fraud-detection.tar.gz "$EC2_USER@$EC2_HOST:~/"
print_info "Image transferred successfully"

# Cleanup local temp file
rm /tmp/fraud-detection.tar.gz

# Step 6: Setup Docker on EC2 (if not installed)
echo ""
echo "Step 6: Setting up Docker on EC2..."
ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" << 'ENDSSH'
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        echo "Installing Docker..."
        sudo apt-get update -qq
        sudo apt-get install -y docker.io > /dev/null
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -aG docker ubuntu
        echo "Docker installed"
    else
        echo "Docker already installed"
    fi
    
    # Install docker-compose if needed
    if ! command -v docker-compose &> /dev/null; then
        echo "Installing docker-compose..."
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
        echo "docker-compose installed"
    fi
ENDSSH
print_info "Docker setup complete"

# Step 7: Load and run Docker container
echo ""
echo "Step 7: Deploying container on EC2..."
ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" << ENDSSH
    # Load Docker image
    echo "Loading Docker image..."
    docker load < ~/fraud-detection.tar.gz
    
    # Verify loaded image architecture
    echo "Verifying image architecture..."
    LOADED_ARCH=\$(docker image inspect $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG 2>/dev/null | grep -o '"Architecture": "[^"]*"' | cut -d'"' -f4)
    echo "Loaded image architecture: \$LOADED_ARCH"
    
    # Stop and remove existing container if exists
    if docker ps -a | grep -q $CONTAINER_NAME; then
        echo "Stopping existing container..."
        docker stop $CONTAINER_NAME || true
        docker rm $CONTAINER_NAME || true
    fi
    
    # Remove old images to free space
    docker image prune -f
    
    # Run new container
    echo "Starting new container..."
    docker run -d \
        --name $CONTAINER_NAME \
        --restart unless-stopped \
        -p $HOST_PORT:$CONTAINER_PORT \
        -e ENVIRONMENT=production \
        $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG
    
    # Wait for container to start
    echo "Waiting for container to start..."
    sleep 10
    
    # Check container status
    if docker ps | grep -q $CONTAINER_NAME; then
        echo "✓ Container is running successfully"
        docker ps | grep $CONTAINER_NAME
        
        # Show logs
        echo ""
        echo "Container logs:"
        docker logs --tail 10 $CONTAINER_NAME
    else
        echo "✗ Container failed to start. Full logs:"
        docker logs $CONTAINER_NAME
        docker ps -a | grep $CONTAINER_NAME
        exit 1
    fi
    
    # Cleanup
    rm ~/fraud-detection.tar.gz
ENDSSH
print_info "Container deployed successfully"

# Step 8: Verify deployment
echo ""
echo "Step 8: Verifying deployment..."
sleep 5

# Test the endpoint from inside the server
echo "Testing from inside server..."
ssh -i "$KEY_FILE" "$EC2_USER@$EC2_HOST" << 'ENDSSH'
    # Test localhost
    if curl -f -s -o /dev/null -w "%{http_code}" http://localhost:80 | grep -q "200"; then
        echo "✓ Localhost:80 responding (200 OK)"
    else
        echo "⚠ Localhost:80 not responding"
        echo "Checking what's listening on port 80:"
        sudo ss -tlnp | grep :80 || echo "Nothing listening on port 80"
    fi
    
    # Check port mapping
    echo ""
    echo "Docker port mapping:"
    docker port fraud-api
ENDSSH

# Test from outside
echo ""
echo "Testing from outside..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://$EC2_HOST/" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" == "200" ]; then
    print_info "External access: Working! (HTTP $HTTP_CODE)"
elif [ "$HTTP_CODE" == "000" ]; then
    print_warning "Cannot reach server externally (connection failed)"
    echo "This usually means port 80 is blocked by AWS Security Group"
else
    print_warning "External access: HTTP $HTTP_CODE"
fi

# Display summary
echo ""
echo "=========================="
echo "✅ DEPLOYMENT COMPLETE!"
echo "=========================="
echo ""
echo "🌐 Your API should be accessible at:"
echo "   http://$EC2_HOST"
echo ""
echo "📋 Useful Commands:"
echo ""
echo "   Connect to server:"
echo "   ssh -i $KEY_FILE $EC2_USER@$EC2_HOST"
echo ""
echo "   View container logs:"
echo "   ssh -i $KEY_FILE $EC2_USER@$EC2_HOST 'docker logs -f $CONTAINER_NAME'"
echo ""
echo "   Check container status:"
echo "   ssh -i $KEY_FILE $EC2_USER@$EC2_HOST 'docker ps'"
echo ""
echo "   Restart container:"
echo "   ssh -i $KEY_FILE $EC2_USER@$EC2_HOST 'docker restart $CONTAINER_NAME'"
echo ""
echo "   Stop container:"
echo "   ssh -i $KEY_FILE $EC2_USER@$EC2_HOST 'docker stop $CONTAINER_NAME'"
echo ""
echo "   Test locally on server:"
echo "   ssh -i $KEY_FILE $EC2_USER@$EC2_HOST 'curl http://localhost:80'"
echo ""

if [ "$HTTP_CODE" != "200" ]; then
    echo "⚠️  IMPORTANT: If external access fails, verify:"
    echo "   1. AWS Security Group has port 80 open for 0.0.0.0/0"
    echo "   2. EC2 instance is running"
    echo "   3. No other service is using port 80"
    echo ""
fi

echo "=========================="