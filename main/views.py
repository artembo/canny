import random
import string

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.shortcuts import render
from django.views.generic import FormView
from skimage.io import imread, imsave

from main.forms import ImageForm
from .canny import CannyEdgeDetector


class ImageView(FormView):
    form_class = ImageForm
    template_name = 'main/index.html'
    orig_image = None

    def form_valid(self, form):
        file = self.request.FILES.get('image').read()
        img = self.orig_image = f'{self.get_rand_path("orig")}'
        default_storage.save(img, ContentFile(file))

        img = imread(f'media/{img}', as_gray=True)

        sigma = form.cleaned_data.get('sigma')
        kernel_size = form.cleaned_data.get('kernel_size')
        weak_pixel = form.cleaned_data.get('weak_pixel')
        strong_pixel = form.cleaned_data.get('strong_pixel')
        low_threshold = form.cleaned_data.get('low_threshold')
        high_threshold = form.cleaned_data.get('high_threshold')
        canny = CannyEdgeDetector(
            img,
            sigma=sigma,
            kernel_size=kernel_size,
            weak_pixel=weak_pixel,
            strong_pixel=strong_pixel,
            low_threshold=low_threshold,
            high_threshold=high_threshold
        )

        images = self.get_images_from_arrays(canny.get_all_stages())

        return render(self.request, template_name='main/index.html', context={'images': images})

    def get_images_from_arrays(self, img_arrays):
        images = [('Оригинал изображения', '/media/' + self.orig_image)]

        for img_array in img_arrays:
            images.append((img_array[0], self.save_image(img_array[1])))

        return images

    def save_image(self, img_array):
        path = f'media/{self.get_rand_path()}'
        imsave(path, img_array)

        return path

    @staticmethod
    def get_rand_path(path='canny_edge'):
        rand = ''.join(random.choice(string.ascii_lowercase) for _ in range(15))
        return f'{path}_{rand}.jpg'
