# 3.11, not 3.10: numpy, pandas, scikit-learn and scipy all declare
# requires-python >= 3.11, so a 3.10 base cannot install requirements.txt.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
