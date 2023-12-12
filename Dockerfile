# Use an official Python runtime as a base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y python3 python3-pip git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Install LinearFold
RUN git clone https://github.com/LinearFold/LinearFold.git \
    && cd LinearFold \
    && make

# Copy the app files into the container
COPY rnatargeting/ ./rnatargeting/
COPY saved_model/ ./saved_model/
COPY .streamlit/ ./.streamlit/
COPY img/ ./img/
COPY app.py ./

# Make port ${PORT} available to the world outside this container
EXPOSE ${PORT}

# Run the app when the container launches
CMD sh -c "streamlit run app.py \
  --server.headless true \
  --server.fileWatcherType none \
  --browser.gatherUsageStats false \
  --server.port=${PORT} \
  --server.address=0.0.0.0"
