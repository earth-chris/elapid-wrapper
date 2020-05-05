FROM conda/miniconda3

# Create environment
COPY environment.yml .
RUN conda env create --file environment.yml
