# Movie Shot Categorizer

A Python library to categorize movie shots using a Hugging Face model.

## Installation

You can install the library directly from the GitHub repository:


```bash
pip install git+https://github.com/your-username/movie-shot-categorizer.git
```

## Use the API 

Now you can use the library in your Python code:

```python
from movie_shot_categorizer import ShotCategorizer

# Initialize the categorizer
categorizer = ShotCategorizer()

# Categorize an image
image_path = "path/to/your/image.jpg" # Replace with the actual image path
categories = categorizer.categorize(image_path)

# Print the results
print(categories)
```

