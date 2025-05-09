# Use the official Python 3.12 image.
FROM mcr.microsoft.com/vscode/devcontainers/python:3.12-bullseye

# Set the working directory in the container
WORKDIR /workspace

# Optional: Install system dependencies if needed
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#    && apt-get -y install --no-install-recommends <your-package-list>

RUN pip install --upgrade pip
RUN pipx install poetry

# Optional: Copy requirements file and install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Keep the container running
CMD ["sleep", "infinity"]
