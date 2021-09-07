FROM pytorch/pytorch

WORKDIR /workspace

ADD . /workspace

RUN pip install -r requirements.txt

CMD [ "python" , "/workspace/app.py" ]

ENV HOME=/workspace
