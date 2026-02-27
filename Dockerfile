FROM python:3.11-slim

WORKDIR /app

# Install requirements first (cached layer)
COPY requirements-live.txt .
RUN pip install --no-cache-dir -r requirements-live.txt

# Copy only what the live runner needs
COPY config/ config/
COPY core/ core/
COPY strategy/ strategy/
COPY live/ live/

# Default: paper trading on IB Gateway
ENV IB_HOST=ib-gateway
ENV IB_PORT=4002

CMD ["python3", "live/ib_runner.py", "--full-session"]
