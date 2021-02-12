# set base image
FROM python:3

# set the working directory in the container
WORKDIR /app

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the content of the local src directory to the working directory
COPY classifiers ./classifiers
COPY docker.py .
COPY download.py .

# install nltk datasets
RUN python download.py

# Create a tmp directory
RUN mkdir tmp
ENV TMPDIR /app/tmp

# command to run on container start
ENTRYPOINT [ "python", "./docker.py" ]