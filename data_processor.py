import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
except:
    pass

class DataProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_skills(self, text):
        """Extract skills from text using common skill patterns"""
        # Common technical skills patterns
        skill_patterns = [
            r'python', r'java', r'javascript', r'c\+\+', r'c#', r'ruby', r'go', r'rust',
            r'sql', r'mysql', r'postgresql', r'mongodb', r'redis',
            r'react', r'angular', r'vue', r'django', r'flask', r'spring',
            r'machine learning', r'deep learning', r'computer vision', r'nlp',
            r'aws', r'azure', r'gcp', r'docker', r'kubernetes', r'jenkins',
            r'tensorflow', r'pytorch', r'scikit-learn', r'pandas', r'numpy',
            r'agile', r'scrum', r'devops', r'ci/cd'
        ]
        
        found_skills = []
        for pattern in skill_patterns:
            if re.search(pattern, text.lower()):
                found_skills.append(pattern)
        
        return found_skills
    
    def load_sample_data(self):
        """Generate sample data for demonstration"""
        # Sample job descriptions
        jobs_data = {
            'job_id': [1, 2, 3, 4, 5],
            'title': [
                'Senior Python Developer',
                'Data Scientist',
                'Frontend Developer',
                'DevOps Engineer',
                'Machine Learning Engineer'
            ],
            'description': [
                'We are looking for a Senior Python Developer with experience in Django, Flask, and REST APIs. Knowledge of PostgreSQL and Docker is required.',
                'Seeking Data Scientist with expertise in machine learning, Python, pandas, and scikit-learn. Experience with TensorFlow and NLP is a plus.',
                'Frontend Developer needed with strong skills in JavaScript, React, HTML5, and CSS3. Experience with Vue.js is desirable.',
                'DevOps Engineer required with AWS, Docker, Kubernetes, and CI/CD pipeline experience. Knowledge of Terraform and Jenkins.',
                'Machine Learning Engineer with deep learning, computer vision, and PyTorch experience. Python programming and model deployment skills.'
            ],
            'required_skills': [
                'python,django,flask,postgresql,docker',
                'python,machine learning,pandas,scikit-learn,tensorflow',
                'javascript,react,html5,css3,vue.js',
                'aws,docker,kubernetes,ci/cd,terraform',
                'python,machine learning,deep learning,pytorch,computer vision'
            ],
            'experience_level': ['Senior', 'Mid', 'Mid', 'Senior', 'Senior']
        }
        
        # Sample candidate profiles
        candidates_data = {
            'candidate_id': [101, 102, 103, 104, 105],
            'name': ['Alice Smith', 'Bob Johnson', 'Carol Davis', 'David Wilson', 'Eva Brown'],
            'skills': [
                'Python, Django, Flask, PostgreSQL, Docker, REST APIs',
                'Python, Machine Learning, pandas, scikit-learn, SQL, Data Analysis',
                'JavaScript, React, HTML5, CSS3, Vue.js, TypeScript',
                'AWS, Docker, Kubernetes, Jenkins, Linux, Bash Scripting',
                'Python, Deep Learning, PyTorch, Computer Vision, TensorFlow, OpenCV'
            ],
            'experience': ['5 years', '3 years', '4 years', '6 years', '4 years'],
            'education': [
                'BS Computer Science',
                'MS Data Science',
                'BS Software Engineering',
                'BS Information Technology',
                'MS Artificial Intelligence'
            ]
        }
        
        jobs_df = pd.DataFrame(jobs_data)
        candidates_df = pd.DataFrame(candidates_data)
        
        return jobs_df, candidates_df