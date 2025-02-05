# Use the rocker/verse image as the base
FROM rocker/verse:latest

# Install Python 3.11 and necessary tools
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev \
    build-essential libyaml-dev && \
    apt-get clean

# Set the working directory
WORKDIR /app

# Create and activate a Python virtual environment with Python 3.11
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install base Python packages first
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir wheel setuptools

# Install core dependencies first with compatible versions
RUN pip3 install --no-cache-dir \
    numpy==1.26.0 \
    scipy==1.10.1 \
    pandas==2.2.3 \
    python-dateutil==2.9.0.post0 \
    pytz==2024.2 \
    six==1.17.0 \
    tzdata==2024.2

# Install scikit-learn first
RUN pip3 install --no-cache-dir scikit-learn==1.3.0

# Install other ML packages separately
RUN pip3 install --no-cache-dir xgboost==1.7.6
RUN pip3 install --no-cache-dir lightgbm==4.0.0
RUN pip3 install --no-cache-dir catboost==1.2

# Install image processing and visualization
RUN pip3 install --no-cache-dir \
    Pillow==10.0.0 \
    matplotlib==3.7.2 \
    opencv-python==4.8.0.76 \
    scikit-image==0.21.0 \
    seaborn==0.12.2

# Install remaining packages
RUN pip3 install --no-cache-dir \
    joblib==1.3.2 \
    scikit-plot==0.3.7 \
    tqdm==4.66.1 \
    pyzbar==0.1.9 \
    scikit-bio==0.5.9 \
    scikit-posthocs==0.7.0

# Copy R requirements file and install R dependencies
COPY r_requirements.txt .
RUN R -e "install.packages('remotes', repos = 'https://cloud.r-project.org')" && \
    R -e "remotes::install_cran(readLines('r_requirements.txt'))"

# Copy the rest of your application code into the container
COPY . .

# Set the default command (modify as needed)
CMD ["tail", "-f", "/dev/null"]