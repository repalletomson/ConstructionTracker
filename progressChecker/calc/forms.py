from django import forms

class ConstructionProgressForm(forms.Form):
    ACTIVITY_CHOICES = [
        ('foundation', 'Foundation'),
        ('super_structure', 'Super Structure'),
        ('furnishing', 'Furnishing'),
        ('interiors', 'Interiors'),
    ]

    image = forms.ImageField(label='Upload Image', required=True)
    activity = forms.ChoiceField(choices=ACTIVITY_CHOICES, label='Select Activity', required=True)
class UploadImageForm(forms.Form):
    image=forms.ImageField()