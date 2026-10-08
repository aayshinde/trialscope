# Dashboard image. It runs from the slim bundle in deploy/ (no pipeline run needed).
#   docker build -t trialscope .  &&  docker run -p 8501:8501 trialscope
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY trialscope/ trialscope/
COPY app.py .
COPY .streamlit/ .streamlit/
COPY deploy/ deploy/
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
