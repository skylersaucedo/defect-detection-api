FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11
WORKDIR /code
COPY . /code
RUN pip install --upgrade pip &&\
   pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip3 install torchvision==0.2.2
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]