# Use an official Python runtime as a parent image
FROM python:3.10
# Set the working directory to /app
WORKDIR /app
# Copy the current directory contents into the container at /app
COPY . /app
# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
# Make port 5000 available to the world outside this container
EXPOSE 5000
ENV NAME venv
# Run the Flask application
CMD ["python", "app.py", "--host=0.0.0.0", "--port=5000"]