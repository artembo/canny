from django import forms


class ImageForm(forms.Form):
    image = forms.ImageField()
    sigma = forms.FloatField(min_value=0.1, max_value=3, initial=1)
    kernel_size = forms.IntegerField(min_value=2, max_value=10, initial=5)
    weak_pixel = forms.IntegerField(min_value=0, max_value=255, initial=75)
    strong_pixel = forms.IntegerField(min_value=0, max_value=255, initial=255)
    low_threshold = forms.DecimalField(min_value=0.01, max_value=1, decimal_places=2, max_digits=3, initial=0.05)
    high_threshold = forms.DecimalField(min_value=0.01, max_value=1, decimal_places=2, max_digits=3, initial=0.15)