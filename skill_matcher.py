import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

class SkillMatcher:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.job_embeddings = None
        self.candidate_embeddings = None
        
    def preprocess_skills(self, skills_text):
        """Preprocess skills text for matching"""
        if pd.isna(skills_text):
            return ""
        return str(skills_text).lower().replace(',', ' ').replace(';', ' ')
    
    def calculate_similarity(self, job_skills, candidate_skills, method='hybrid'):
        """Calculate similarity between job and candidate skills"""
        job_skills_clean = self.preprocess_skills(job_skills)
        candidate_skills_clean = self.preprocess_skills(candidate_skills)
        
        if method == 'tfidf':
            return self._tfidf_similarity(job_skills_clean, candidate_skills_clean)
        elif method == 'sentence':
            return self._sentence_similarity(job_skills_clean, candidate_skills_clean)
        else:  # hybrid
            tfidf_sim = self._tfidf_similarity(job_skills_clean, candidate_skills_clean)
            sentence_sim = self._sentence_similarity(job_skills_clean, candidate_skills_clean)
            return (tfidf_sim + sentence_sim) / 2
    
    def _tfidf_similarity(self, text1, text2):
        """Calculate TF-IDF cosine similarity"""
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return similarity[0][0]
        except:
            return 0.0
    
    def _sentence_similarity(self, text1, text2):
        """Calculate sentence embedding similarity"""
        try:
            embeddings = self.sentence_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
            return similarity[0][0]
        except:
            return 0.0
    
    def match_candidates_to_job(self, job_description, candidates_df, top_n=5):
        """Match candidates to a specific job"""
        similarities = []
        
        for _, candidate in candidates_df.iterrows():
            similarity_score = self.calculate_similarity(
                job_description, 
                candidate['skills']
            )
            similarities.append(similarity_score)
        
        candidates_df = candidates_df.copy()
        candidates_df['match_score'] = similarities
        candidates_df['match_percentage'] = (candidates_df['match_score'] * 100).round(2)
        
        # Sort by match score and return top matches
        matched_candidates = candidates_df.sort_values('match_percentage', ascending=False).head(top_n)
        
        return matched_candidates
    
    def find_jobs_for_candidate(self, candidate_skills, jobs_df, top_n=5):
        """Find suitable jobs for a specific candidate"""
        similarities = []
        
        for _, job in jobs_df.iterrows():
            similarity_score = self.calculate_similarity(
                job['required_skills'],
                candidate_skills
            )
            similarities.append(similarity_score)
        
        jobs_df = jobs_df.copy()
        jobs_df['match_score'] = similarities
        jobs_df['match_percentage'] = (jobs_df['match_score'] * 100).round(2)
        
        # Sort by match score and return top matches
        matched_jobs = jobs_df.sort_values('match_percentage', ascending=False).head(top_n)
        
        return matched_jobs
    
    def skill_gap_analysis(self, job_skills, candidate_skills):
        """Analyze skill gaps between job requirements and candidate skills"""
        job_skills_set = set(self.preprocess_skills(job_skills).split())
        candidate_skills_set = set(self.preprocess_skills(candidate_skills).split())
        
        missing_skills = job_skills_set - candidate_skills_set
        matching_skills = job_skills_set & candidate_skills_set
        
        return {
            'matching_skills': list(matching_skills),
            'missing_skills': list(missing_skills),
            'match_percentage': len(matching_skills) / len(job_skills_set) * 100 if job_skills_set else 0
        }