FROM python:3.10.13-slim

# Install system dependencies
# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user "squid-user"
RUN addgroup --system squid-user && adduser --system --ingroup squid-user squid-user

RUN pip install --upgrade pip

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app with proper ownership
COPY . .

RUN mkdir -p /app/logs && \
    chmod 777 /app /app/logs && \
    chown -R squid-user:squid-user /app /app/logs

# Add /app to the list of safe directories for Git
RUN git config --global --add safe.directory /app

# Remove all files matching .gitignore patterns and .git directory
RUN git clean -fdX && rm -rf .git

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install .

# Diagnostic steps:
RUN echo "Listing site-packages to check for squid_control installation:" && \
    pip show squid_control && \
    ls -R $(python -c "import site; print(site.getsitepackages()[0])")/squid_control && \
    echo "Attempting to import hypha_tools from squid_control directly:" && \
    python -c "from squid_control.hypha_tools.artifact_manager import artifact_manager; print('Successfully imported artifact_manager')"

# Run the application as the non-root user "squid-user"
USER squid-user
CMD ["python", "start_hypha_service.py", "--simulation"]