# Anomaly detection on Kairos Web Interface
The base setup of this project has been taken from the following repository:
- https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-clinical-analytics

This web interface in written using python3.8 with the library Dash. It is used to visualize the data collected by the anomaly detection scripts.

# Installation
- Install python3.8 (conda or virtualenv)
- Install the requirements.txt in your python3.8 environment
- Now to run the web interface you can use:
- 
```bash
python app.py 
```

# Build the docker image
```bash
docker build -t anomalydetectionkairos_web .
```
Now you can run the docker image with:
```bash
# name the container anomalydetectionkairos_web
docker run -p 8080:8080 --name anomalydetectionkairos_web anomalydetectionkairos_web  
```