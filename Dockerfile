FROM python:3.7

RUN pip install django

WORKDIR /web

CMD python manage.py runserver
