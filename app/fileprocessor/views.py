import json
from django.http import JsonResponse
from django.shortcuts import render
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from .forms import CSVUploadForm
from .models import CSVFile

class SPAView(View):
    def get(self, request):
        form = CSVUploadForm()
        return render(request, 'fileprocessor/spa.html', {'form': form})


@method_decorator(csrf_exempt, name='dispatch')
class UploadCSVView(View):
    def post(self, request):
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = form.save()
            return JsonResponse({
                'success': True,
                'file_id': str(csv_file.id),
                'message': 'File uploaded successfully'
            })
        else:
            errors = {field: str(error[0]) for field, error in form.errors.items()}
            return JsonResponse({
                'success': False,
                'errors': errors
            })


# @method_decorator(csrf_exempt, name='dispatch-download')
# class DownloadCSVView(View):
#     def get(self, request):
#         form = CSVUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             csv_file = form.save()
#             return JsonResponse({
#                 'success': True,
#                 'file_id': str(csv_file.id),
#                 'message': 'File uploaded successfully'
#             })
#         else:
#             errors = {field: str(error[0]) for field, error in form.errors.items()}
#             return JsonResponse({
#                 'success': False,
#                 'errors': errors
#             })

