# Fraud Detection API - Docker Setup

This guide will help you package and run the fraud detection model as a Docker container with a FastAPI REST API.

## Quick Start

### 1. Build the Docker Image

```bash
# Build the Docker image
docker build -t fraud-detection-api .

# Or using docker-compose
docker-compose build
```

### 2. Run the Container

```bash
# Run the container
docker run -p 8000:8000 fraud-detection-api

# Or using docker-compose
docker-compose up
```

### 3. Test the API

Once the container is running, you can access:

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Root Endpoint**: http://localhost:8000/

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict?threshold=0.5" \
     -H "Content-Type: application/json" \
     -d '{
       "V1": -0.7059898246110177,
       "V2": 0.6277668093643811,
       "V3": -0.035994995232166,
       "V4": 0.1806427850874308,
       "V5": 0.4599348239833234,
       "V6": -0.036283158251373,
       "V7": 0.2802046719288935,
       "V8": -0.1841152764576969,
       "V9": 0.0685241005919484,
       "V10": 0.5863629005107058,
       "V11": -0.25233334795008,
       "V12": -1.2299078418984513,
       "V13": 0.4682882741114543,
       "V14": 0.4017355215141967,
       "V15": -0.3078030347127327,
       "V16": -0.1123814085906342,
       "V17": -0.4589679556521681,
       "V18": 0.0405522364190535,
       "V19": -0.9375302972907276,
       "V20": 0.1741002550832633,
       "V21": -0.1256561406066695,
       "V22": -0.1784533927889745,
       "V23": -0.1156088530642112,
       "V24": -0.2434813742463694,
       "V25": -1.156796820313679,
       "V26": 1.148810949147973,
       "V27": 1.0191007119749338,
       "V28": 0.0030985451139533,
       "Time": 0.0037648659393779,
       "Amount": -0.307400143722893
     }'
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict_batch?threshold=0.5" \
     -H "Content-Type: application/json" \
     -d '[
       {
         "V1": -0.7059898246110177,
         "V2": 0.6277668093643811,
         "V3": -0.035994995232166,
         "V4": 0.1806427850874308,
         "V5": 0.4599348239833234,
         "V6": -0.036283158251373,
         "V7": 0.2802046719288935,
         "V8": -0.1841152764576969,
         "V9": 0.0685241005919484,
         "V10": 0.5863629005107058,
         "V11": -0.25233334795008,
         "V12": -1.2299078418984513,
         "V13": 0.4682882741114543,
         "V14": 0.4017355215141967,
         "V15": -0.3078030347127327,
         "V16": -0.1123814085906342,
         "V17": -0.4589679556521681,
         "V18": 0.0405522364190535,
         "V19": -0.9375302972907276,
         "V20": 0.1741002550832633,
         "V21": -0.1256561406066695,
         "V22": -0.1784533927889745,
         "V23": -0.1156088530642112,
         "V24": -0.2434813742463694,
         "V25": -1.156796820313679,
         "V26": 1.148810949147973,
         "V27": 1.0191007119749338,
         "V28": 0.0030985451139533,
         "Time": 0.0037648659393779,
         "Amount": -0.307400143722893
       }
     ]'
```

### Model Information
```bash
curl http://localhost:8000/model_info
```

## Configuration

### Environment Variables

- `PYTHONPATH`: Set to `/app` (already configured in Dockerfile)
- `PYTHONDONTWRITEBYTECODE`: Prevents Python from writing .pyc files
- `PYTHONUNBUFFERED`: Ensures Python output is sent straight to terminal

### Threshold Configuration

The fraud detection threshold can be configured per request:

- **Default threshold**: 0.5
- **Range**: 0.0 to 1.0
- **Higher threshold**: More conservative (fewer fraud predictions)
- **Lower threshold**: More sensitive (more fraud predictions)

## Docker Commands Reference

### Build Commands
```bash
# Build with specific tag
docker build -t fraud-detection-api:latest .

# Build without cache
docker build --no-cache -t fraud-detection-api .

# Build with specific Dockerfile
docker build -f Dockerfile -t fraud-detection-api .
```

### Run Commands
```bash
# Run in foreground
docker run -p 8000:8000 fraud-detection-api

# Run in background
docker run -d -p 8000:8000 --name fraud-api fraud-detection-api

# Run with custom port
docker run -p 8080:8000 fraud-detection-api

# Run with volume mount for model updates
docker run -p 8000:8000 -v $(pwd)/models:/app/models:ro fraud-detection-api
```

### Management Commands
```bash
# View running containers
docker ps

# View logs
docker logs fraud-api

# Stop container
docker stop fraud-api

# Remove container
docker rm fraud-api

# Remove image
docker rmi fraud-detection-api
```

### Docker Compose Commands
```bash
# Start services
docker-compose up

# Start in background
docker-compose up -d

# Stop services
docker-compose down

# Rebuild and start
docker-compose up --build

# View logs
docker-compose logs -f
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Use a different port
   docker run -p 8080:8000 fraud-detection-api
   ```

2. **Model not found**
   - Ensure the `models/` directory contains the trained model files
   - Check that the model files are copied correctly in the Dockerfile

3. **Permission denied**
   - The Dockerfile creates a non-root user for security
   - Ensure the models directory has proper permissions

4. **Memory issues**
   - Increase Docker memory limit if needed
   - Consider using a smaller base image

### Debugging

```bash
# Run container interactively
docker run -it fraud-detection-api /bin/bash

# Check container logs
docker logs <container_id>

# Inspect container
docker inspect <container_id>

# Check health status
curl http://localhost:8000/health
```

## Production Considerations

### Security
- The container runs as a non-root user
- Only necessary ports are exposed
- Health checks are configured

### Performance
- Uses uvicorn with standard workers
- Consider using gunicorn for production
- Monitor memory usage and adjust accordingly

### Monitoring
- Health check endpoint available
- Structured logging implemented
- Consider adding metrics collection

### Scaling
- Use docker-compose for multiple instances
- Consider using Kubernetes for orchestration
- Implement load balancing for high availability

## Model Updates

To update the model without rebuilding the container:

1. Replace model files in the `models/` directory
2. Restart the container:
   ```bash
   docker-compose restart
   ```

Or use volume mounting for automatic updates:
```bash
docker run -p 8000:8000 -v $(pwd)/models:/app/models:ro fraud-detection-api
```
