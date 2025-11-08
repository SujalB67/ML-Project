import gradio as gr
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class EnhancedSkillMatcher:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Enhanced skill taxonomy
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust', 'typescript'],
            'web_frameworks': ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'node', 'express'],
            'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sqlite'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform'],
            'data_science': ['machine learning', 'deep learning', 'computer vision', 'nlp', 'tensorflow', 'pytorch'],
            'data_tools': ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn'],
            'devops': ['ci/cd', 'git', 'linux', 'bash', 'ansible', 'prometheus'],
            'soft_skills': ['communication', 'leadership', 'problem solving', 'teamwork', 'agile', 'scrum']
        }

        self.all_skills = [skill for category in self.skill_categories.values() for skill in category]

    def extract_skills_from_text(self, text):
        """Enhanced skill extraction with categories"""
        if pd.isna(text):
            return []

        text_lower = str(text).lower()
        found_skills = []
        skill_categories_found = {}

        for category, skills in self.skill_categories.items():
            category_skills = []
            for skill in skills:
                if skill in text_lower:
                    category_skills.append(skill)
                    found_skills.append(skill)
            if category_skills:
                skill_categories_found[category] = category_skills

        return found_skills, skill_categories_found

    def calculate_enhanced_similarity(self, job_text, candidate_text):
        """Calculate similarity with skill category weighting"""
        # Basic text similarity
        job_skills, job_categories = self.extract_skills_from_text(job_text)
        candidate_skills, candidate_categories = self.extract_skills_from_text(candidate_text)

        if not job_skills:
            return 0.0, {}, {}

        # Calculate overlap
        matching_skills = set(job_skills) & set(candidate_skills)
        missing_skills = set(job_skills) - set(candidate_skills)

        # Base match percentage
        base_match = len(matching_skills) / len(job_skills) * 100 if job_skills else 0

        # Category-based weighting
        category_scores = {}
        for category in self.skill_categories.keys():
            job_cat_skills = set(job_categories.get(category, []))
            candidate_cat_skills = set(candidate_categories.get(category, []))

            if job_cat_skills:
                cat_match = len(job_cat_skills & candidate_cat_skills) / len(job_cat_skills) * 100
                category_scores[category] = cat_match

        # Weighted final score (you can adjust weights based on job role)
        final_score = base_match

        return final_score, matching_skills, missing_skills

class SkillSyncAI:
    def __init__(self):
        self.skill_matcher = EnhancedSkillMatcher()
        self.jobs_df, self.candidates_df = self.load_enhanced_sample_data()

    def load_enhanced_sample_data(self):
        """Enhanced sample data with more skills"""
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
                'Looking for Python expert with Django/Flask experience. Must know PostgreSQL, Docker, and REST APIs. AWS knowledge preferred.',
                'Seeking Data Scientist with strong ML background. Python, pandas, scikit-learn required. TensorFlow/PyTorch experience needed.',
                'Frontend Developer with React, JavaScript, HTML5, CSS3. Vue.js and TypeScript are nice to have.',
                'DevOps Engineer with AWS, Docker, Kubernetes expertise. CI/CD pipelines and Terraform experience required.',
                'ML Engineer with deep learning and computer vision experience. PyTorch, TensorFlow, and Python proficiency needed.'
            ],
            'required_skills': [
                'python,django,flask,postgresql,docker,rest,aws',
                'python,machine learning,pandas,scikit-learn,tensorflow,pytorch,sql',
                'javascript,react,html5,css3,vue.js,typescript',
                'aws,docker,kubernetes,ci/cd,terraform,linux',
                'python,machine learning,deep learning,pytorch,tensorflow,computer vision'
            ],
            'experience_level': ['Senior', 'Mid', 'Mid', 'Senior', 'Senior'],
            'salary_range': ['$120k-$150k', '$100k-$130k', '$90k-$120k', '$110k-$140k', '$130k-$160k']
        }

        candidates_data = {
            'candidate_id': [101, 102, 103, 104, 105],
            'name': ['Alice Smith', 'Bob Johnson', 'Carol Davis', 'David Wilson', 'Eva Brown'],
            'skills': [
                'Python, Django, Flask, PostgreSQL, Docker, REST APIs, AWS, JavaScript',
                'Python, Machine Learning, pandas, scikit-learn, SQL, Data Analysis, Statistics',
                'JavaScript, React, HTML5, CSS3, Vue.js, TypeScript, Node.js',
                'AWS, Docker, Kubernetes, Jenkins, Linux, Bash, Terraform, Python',
                'Python, Deep Learning, PyTorch, Computer Vision, TensorFlow, OpenCV, numpy'
            ],
            'experience': ['5 years', '3 years', '4 years', '6 years', '4 years'],
            'education': [
                'BS Computer Science',
                'MS Data Science',
                'BS Software Engineering',
                'BS Information Technology',
                'MS Artificial Intelligence'
            ],
            'current_role': [
                'Python Developer',
                'Junior Data Scientist',
                'Frontend Developer',
                'DevOps Engineer',
                'ML Researcher'
            ]
        }

        return pd.DataFrame(jobs_data), pd.DataFrame(candidates_data)

    def create_interface(self):
        with gr.Blocks(
            title="🎯 SkillSync AI - Where Skills Meet Opportunities",
            theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
        ) as demo:

            # Header
            gr.Markdown(
                """
                # 🎯 SkillSync AI
                ### *Where skills meet opportunities*
                **This tool uses advanced AI to match job seekers with suitable roles and vice versa.**
                """
            )

            with gr.Tab("🔍 Match Candidates to Job"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Job Requirements")
                        job_dropdown = gr.Dropdown(
                            choices=[f"{row['title']} (ID: {row['job_id']})" for _, row in self.jobs_df.iterrows()],
                            label="Select Job Position",
                            value=f"{self.jobs_df.iloc[0]['title']} (ID: {self.jobs_df.iloc[0]['job_id']})"
                        )
                        job_description = gr.Textbox(
                            label="Job Description",
                            lines=3,
                            interactive=True
                        )
                        job_skills_display = gr.Textbox(
                            label="Required Skills",
                            interactive=False
                        )
                        top_n_candidates = gr.Slider(
                            minimum=1, maximum=10, value=5,
                            label="Number of Top Matches to Show"
                        )
                        match_candidates_btn = gr.Button(
                            "🚀 Find Matching Candidates",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Matching Results")
                        candidates_output = gr.Dataframe(
                            label="Top Matching Candidates",
                            headers=["ID", "Name", "Skills", "Experience", "Match %", "Current Role"],
                            datatype=["number", "str", "str", "str", "number", "str"],
                            interactive=False
                        )
                        match_plot = gr.Plot(label="Match Scores Visualization")

            with gr.Tab("💼 Find Jobs for Candidate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Candidate Profile")
                        candidate_dropdown = gr.Dropdown(
                            choices=[f"{row['name']} (ID: {row['candidate_id']})" for _, row in self.candidates_df.iterrows()],
                            label="Select Candidate",
                            value=f"{self.candidates_df.iloc[0]['name']} (ID: {self.candidates_df.iloc[0]['candidate_id']})"
                        )
                        candidate_skills = gr.Textbox(
                            label="Candidate Skills",
                            lines=3,
                            interactive=True
                        )
                        candidate_info = gr.Textbox(
                            label="Candidate Info",
                            interactive=False
                        )
                        top_n_jobs = gr.Slider(
                            minimum=1, maximum=10, value=5,
                            label="Number of Top Jobs to Show"
                        )
                        match_jobs_btn = gr.Button(
                            "🚀 Find Suitable Jobs",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Job Recommendations")
                        jobs_output = gr.Dataframe(
                            label="Top Matching Jobs",
                            headers=["ID", "Title", "Required Skills", "Level", "Salary", "Match %"],
                            datatype=["number", "str", "str", "str", "str", "number"],
                            interactive=False
                        )
                        jobs_plot = gr.Plot(label="Job Match Visualization")

            with gr.Tab("📊 Skill Gap Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Compare Skills")
                        gap_job_select = gr.Dropdown(
                            choices=[f"{row['title']} (ID: {row['job_id']})" for _, row in self.jobs_df.iterrows()],
                            label="Select Job Position",
                            value=f"{self.jobs_df.iloc[0]['title']} (ID: {self.jobs_df.iloc[0]['job_id']})"
                        )
                        gap_candidate_select = gr.Dropdown(
                            choices=[f"{row['name']} (ID: {row['candidate_id']})" for _, row in self.candidates_df.iterrows()],
                            label="Select Candidate",
                            value=f"{self.candidates_df.iloc[1]['name']} (ID: {self.candidates_df.iloc[1]['candidate_id']})"
                        )
                        analyze_gap_btn = gr.Button(
                            "📈 Analyze Skill Gap",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Analysis Results")
                        with gr.Row():
                            with gr.Column():
                                match_percentage_display = gr.Number(
                                    label="Overall Match Percentage",
                                    precision=1
                                )
                            with gr.Column():
                                matching_skills_count = gr.Number(
                                    label="Matching Skills",
                                    precision=0
                                )
                            with gr.Column():
                                missing_skills_count = gr.Number(
                                    label="Skills to Develop",
                                    precision=0
                                )

                        with gr.Row():
                            with gr.Column():
                                matching_skills_list = gr.HighlightedText(
                                    label="✅ Matching Skills",
                                    show_label=True
                                )
                            with gr.Column():
                                missing_skills_list = gr.HighlightedText(
                                    label="📚 Skills to Develop",
                                    show_label=True
                                )

                        skills_radar = gr.Plot(label="Skills Radar Chart")
                        skills_bar_chart = gr.Plot(label="Skills Comparison")

            # Event handlers
            job_dropdown.change(
                self.update_job_info,
                inputs=[job_dropdown],
                outputs=[job_description, job_skills_display]
            )

            candidate_dropdown.change(
                self.update_candidate_info,
                inputs=[candidate_dropdown],
                outputs=[candidate_skills, candidate_info]
            )

            match_candidates_btn.click(
                self.find_matching_candidates,
                inputs=[job_dropdown, top_n_candidates],
                outputs=[candidates_output, match_plot]
            )

            match_jobs_btn.click(
                self.find_matching_jobs,
                inputs=[candidate_dropdown, top_n_jobs],
                outputs=[jobs_output, jobs_plot]
            )

            analyze_gap_btn.click(
                self.enhanced_skill_gap_analysis,
                inputs=[gap_job_select, gap_candidate_select],
                outputs=[
                    match_percentage_display, matching_skills_count, missing_skills_count,
                    matching_skills_list, missing_skills_list, skills_radar, skills_bar_chart
                ]
            )

            # Initialize
            demo.load(
                self.initialize_values,
                outputs=[job_description, job_skills_display, candidate_skills, candidate_info]
            )

        return demo

    def initialize_values(self):
        first_job = self.jobs_df.iloc[0]
        first_candidate = self.candidates_df.iloc[0]

        job_desc = first_job['description']
        job_skills = first_job['required_skills']
        candidate_skills = first_candidate['skills']
        candidate_info = f"{first_candidate['current_role']} | {first_candidate['experience']} | {first_candidate['education']}"

        return job_desc, job_skills, candidate_skills, candidate_info

    def update_job_info(self, job_selection):
        job_id = int(job_selection.split("ID: ")[1].strip(")"))
        job_data = self.jobs_df[self.jobs_df['job_id'] == job_id].iloc[0]
        return job_data['description'], job_data['required_skills']

    def update_candidate_info(self, candidate_selection):
        candidate_id = int(candidate_selection.split("ID: ")[1].strip(")"))
        candidate_data = self.candidates_df[self.candidates_df['candidate_id'] == candidate_id].iloc[0]
        candidate_info = f"{candidate_data['current_role']} | {candidate_data['experience']} | {candidate_data['education']}"
        return candidate_data['skills'], candidate_info

    def find_matching_candidates(self, job_selection, top_n):
        job_id = int(job_selection.split("ID: ")[1].strip(")"))
        job_data = self.jobs_df[self.jobs_df['job_id'] == job_id].iloc[0]

        matches = []
        for _, candidate in self.candidates_df.iterrows():
            match_score, matching_skills, missing_skills = self.skill_matcher.calculate_enhanced_similarity(
                job_data['required_skills'], candidate['skills']
            )
            matches.append({
                'candidate_id': candidate['candidate_id'],
                'name': candidate['name'],
                'skills': candidate['skills'],
                'experience': candidate['experience'],
                'match_percentage': round(match_score, 1),
                'current_role': candidate['current_role']
            })

        matches_df = pd.DataFrame(matches)
        matches_df = matches_df.sort_values('match_percentage', ascending=False).head(top_n)

        # Create visualization
        fig = px.bar(
            matches_df,
            x='name',
            y='match_percentage',
            title=f"🏆 Top {top_n} Candidate Matches for {job_data['title']}",
            labels={'name': 'Candidate', 'match_percentage': 'Match Percentage (%)'},
            color='match_percentage',
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        return matches_df, fig

    def find_matching_jobs(self, candidate_selection, top_n):
        candidate_id = int(candidate_selection.split("ID: ")[1].strip(")"))
        candidate_data = self.candidates_df[self.candidates_df['candidate_id'] == candidate_id].iloc[0]

        matches = []
        for _, job in self.jobs_df.iterrows():
            match_score, matching_skills, missing_skills = self.skill_matcher.calculate_enhanced_similarity(
                job['required_skills'], candidate_data['skills']
            )
            matches.append({
                'job_id': job['job_id'],
                'title': job['title'],
                'required_skills': job['required_skills'],
                'experience_level': job['experience_level'],
                'salary_range': job['salary_range'],
                'match_percentage': round(match_score, 1)
            })

        matches_df = pd.DataFrame(matches)
        matches_df = matches_df.sort_values('match_percentage', ascending=False).head(top_n)

        # Create visualization
        fig = px.bar(
            matches_df,
            x='title',
            y='match_percentage',
            title=f"💼 Top {top_n} Job Matches for {candidate_data['name']}",
            labels={'title': 'Job Title', 'match_percentage': 'Match Percentage (%)'},
            color='match_percentage',
            color_continuous_scale='plasma'
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        return matches_df, fig

    def enhanced_skill_gap_analysis(self, job_selection, candidate_selection):
        job_id = int(job_selection.split("ID: ")[1].strip(")"))
        candidate_id = int(candidate_selection.split("ID: ")[1].strip(")"))

        job_data = self.jobs_df[self.jobs_df['job_id'] == job_id].iloc[0]
        candidate_data = self.candidates_df[self.candidates_df['candidate_id'] == candidate_id].iloc[0]

        match_percentage, matching_skills, missing_skills = self.skill_matcher.calculate_enhanced_similarity(
            job_data['required_skills'], candidate_data['skills']
        )

        # Prepare highlighted text components
        matching_text = [(skill, "match") for skill in matching_skills]
        missing_text = [(skill, "missing") for skill in missing_skills]

        # Create radar chart
        radar_fig = self.create_enhanced_radar_chart(
            list(matching_skills),
            list(missing_skills),
            job_data['title'],
            candidate_data['name']
        )

        # Create bar chart
        bar_fig = self.create_skills_bar_chart(
            list(matching_skills),
            list(missing_skills)
        )

        return (
            match_percentage,
            len(matching_skills),
            len(missing_skills),
            matching_text,
            missing_text,
            radar_fig,
            bar_fig
        )

    def create_enhanced_radar_chart(self, matching_skills, missing_skills, job_title, candidate_name):
        categories = ['Technical Fit', 'Skill Coverage', 'Role Alignment', 'Growth Potential', 'Overall Match']

        # Calculate metrics
        total_required = len(matching_skills) + len(missing_skills)
        technical_fit = len(matching_skills) / total_required if total_required > 0 else 0
        skill_coverage = min(1.0, len(matching_skills) / 8)  # Normalize
        role_alignment = technical_fit * 0.8 + skill_coverage * 0.2
        growth_potential = 1 - (len(missing_skills) / max(1, total_required))
        overall_match = (technical_fit + skill_coverage + role_alignment + growth_potential) / 4

        values = [technical_fit, skill_coverage, role_alignment, growth_potential, overall_match]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=f'{candidate_name}',
            line=dict(color='#4361ee', width=2),
            fillcolor='rgba(67, 97, 238, 0.3)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(tickfont=dict(size=11))
            ),
            showlegend=True,
            title=f"📊 Skills Analysis: {candidate_name} vs {job_title}",
            font=dict(size=12),
            height=400
        )

        return fig

    def create_skills_bar_chart(self, matching_skills, missing_skills):
        categories = ['Matching Skills', 'Skills to Develop']
        values = [len(matching_skills), len(missing_skills)]
        colors = ['#4CAF50', '#FF6B6B']

        fig = px.bar(
            x=categories,
            y=values,
            title="📈 Skills Breakdown",
            labels={'x': 'Category', 'y': 'Number of Skills'},
            color=categories,
            color_discrete_sequence=colors
        )

        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=300
        )

        return fig

# Run the application
def main():
    app = SkillSyncAI()
    demo = app.create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()