FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11
WORKDIR /code
COPY . /code
RUN pip install --upgrade pip &&\
   pip install --no-cache-dir --upgrade -r /code/requirements.txt


RUN pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]