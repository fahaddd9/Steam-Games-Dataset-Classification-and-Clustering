from flask import Flask, render_template
import os

app = Flask(__name__)

# Helper function to read markdown files
def read_markdown(file_path):
    with open(file_path, 'r') as f:
        return f.read()

# Routes for each page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dataset')
def dataset():
    # Preprocessing results adjusted for the updated dataset.html
    preprocessing_results = {
        "dataset_shape": "Dataset Shape: (40833, 20)",
        "columns": [
            'url', 'types', 'name', 'desc_snippet', 'recent_reviews', 'all_reviews', 
            'release_date', 'developer', 'publisher', 'popular_tags', 'game_details', 
            'languages', 'achievements', 'genre', 'game_description', 'mature_content', 
            'minimum_requirements', 'recommended_requirements', 'original_price', 'discount_price'
        ],
        "genre_distribution": """Action                                                         2386
Action,Indie                                                   2129
                                                               1520
                                                               ...
Casual,Indie,Massively Multiplayer,Sports                         1
Design & Illustration,Game Development                            1
Design & Illustration,Utilities,Early Access                      1
Adventure,Casual,Indie,RPG,Simulation,Strategy,Early Access       1
Casual,Racing,Simulation,Sports                                   1
Name: count, Length: 1768, dtype: int64""",
        "filtered_genre_distribution": {
            "Action": 16290,
            "Adventure": 6854,
            "Simulation": 2074,
            "RPG": 927
        },
        "completion_message": "Preprocessing completed. Data saved for classification and clustering."
    }
    return render_template('dataset.html', preprocessing=preprocessing_results)

@app.route('/classification')
def classification():
    performance_report = read_markdown('performance_report.md')
    return render_template('classification.html', performance_report=performance_report)

@app.route('/clustering')
def clustering():
    return render_template('clustering.html')

@app.route('/improvements')
def improvements():
    improvement_report = read_markdown('improvement_report.md')
    return render_template('improvements.html', improvement_report=improvement_report)

@app.route('/comparison')
def comparison():
    comparison_report = read_markdown('comparison_report.md')
    return render_template('comparison.html', comparison_report=comparison_report)

if __name__ == '__main__':
    app.run(debug=True)