FROM python:3.9-slim

RUN mkdir /app
COPY . /app
WORKDIR /app

RUN pip install streamlit 
RUN pip install pandas 
RUN pip install numpy 

EXPOSE 8507

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "model.py", "--server.port=8507", "--server.address=0.0.0.0"]