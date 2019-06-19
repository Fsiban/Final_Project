FROM ubuntu:latest

RUN apt-get update

RUN apt-get -y install python3 \
    && apt-get install -y python3-pip \
    && pip3 install --upgrade pip

RUN pip3 install numpy pandas argparse matplotlib seaborn plotly sklearn

COPY final-project-code.py .

COPY mean_radius_distribution.png .

COPY pairplot.png .

CMD ["python3","-u","final-project-code.py"]

