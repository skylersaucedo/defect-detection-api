FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11
WORKDIR /code
COPY . /code
RUN pip install --upgrade pip &&\
   pip install --no-cache-dir --upgrade -r /code/requirements.txt


FROM nvcr.io/nvidia/pytorch:21.09-py3

COPY . odtk/
RUN pip install --no-cache-dir -e odtk/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]