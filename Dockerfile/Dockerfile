# Build from the repo root: docker build -f Dockerfile/Dockerfile -t agri-price .
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system-level dependencies required by some ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire project
COPY . .

# Train the default (Carrot) model. Use train_all.py instead to bake in every
# commodity, at the cost of a longer build.
RUN python TrainModel/trainmodel.py --commodity Carrot

EXPOSE 8501

# Set the command to run the Streamlit dashboard
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]