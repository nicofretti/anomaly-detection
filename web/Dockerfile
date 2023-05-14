FROM python:3.10
COPY requirements.txt ./requirements.txt
RUN cat requirements.txt
RUN pip install -r requirements.txt
#RUN pip install gunicorn==20.1.0
COPY . ./
EXPOSE 8080
CMD python app.py