# Use an official Python runtime as a parent image
FROM python:3.10.8

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py", "--host=0.0.0.0", "--port=5000"]
# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.2/containers/codespaces-linux/.devcontainer/base.Dockerfile

#FROM mcr.microsoft.com/vscode/devcontainers/universal:2-focal

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
#RUN apt-get update && apt-get -y install --no-install-recommends \
 #  python3.8-venv \
  # gcc 

#ARG USER="codespace"
#ARG VENV_PATH="/home/${USER}/venv"
#COPY ../data_pipe/requirements.txt /tmp/
#COPY ../data_pipe/Makefile /tmp/
#RUN su $USER -c "/usr/bin/python3 -m venv /home/${USER}/venv" \
 #  && su $USER -c "${VENV_PATH}/bin/pip --disable-pip-version-check --no-cache-dir install -r /tmp/requirements.txt" \
  # && rm -rf /tmp/requirements.txt 
