FROM python:3.10

WORKDIR /app
COPY ./ .
RUN pip install .

ENTRYPOINT ["streamlit", "run", "/app/app/Home.py"]