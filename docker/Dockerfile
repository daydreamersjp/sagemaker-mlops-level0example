FROM public.ecr.aws/lambda/python:3.6 AS build
LABEL maintainer="Motoharu Dei"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt --target "${LAMBDA_TASK_ROOT}"

FROM build
COPY test.csv ${LAMBDA_TASK_ROOT}
COPY app.py ${LAMBDA_TASK_ROOT}
CMD [ "app.handler" ]

    
