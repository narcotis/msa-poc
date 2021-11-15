from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
# Create your views here.

def get(request):
    if request.method != "GET":
        return JsonResponse({"status": "error"})

    headers = dict(request.headers)
    cookies = dict(request.COOKIES)

    return JsonResponse({
        "headers": headers,
        "COOKIES": cookies
    })

@csrf_exempt
def post(request):
    if request.method != "POST":
        return JsonResponse({"status": "error"})

    headers = dict(request.headers)
    body = json.loads(request.body)
    cookies = dict(request.COOKIES)
    return JsonResponse({
        "headers": headers,
        "body": body,
        "COOKIES": cookies
    })

def auth(request):
    if request.method != "GET":
        return JsonResponse({"status": "error"})

    return JsonResponse({"status": "success"})