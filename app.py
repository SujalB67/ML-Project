import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_processor import DataProcessor
from skill_matcher import SkillMatcher

class SkillMatcherApp:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.skill_matcher = SkillMatcher()
        self.jobs_df, self.candidates_df = self.data_processor.load_sample_data()
        
    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="AI-Powered Skill Matcher", theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                """
                # 🎯 AI-Powered Skill Matcher
                ### Bridging the Gap Between Job Seekers and Roles
                
                This tool uses advanced AI to match job seekers with suitable roles and vice versa.
                """
            )
            
            with gr.Tab("🔍 Match Candidates to Job"):
                with gr.Row():
                    with gr.Column():
                        job_dropdown = gr.Dropdown(
                            choices=[f"{row['title']} (ID: {row['job_id']})" for _, row in self.jobs_df.iterrows()],
                            label="Select Job Position",
                            value=f"{self.jobs_df.iloc[0]['title']} (ID: {self.jobs_df.iloc[0]['job_id']})"
                        )
                        
                        job_description = gr.Textbox(
                            label="Job Description",
                            lines=4,
                            interactive=True
                        )
                        
                        top_n_candidates = gr.Slider(
                            minimum=1, maximum=10, value=5,
                            label="Number of Top Matches to Show"
                        )
                        
                        match_candidates_btn = gr.Button("Find Matching Candidates", variant="primary")
                    
                    with gr.Column():
                        candidates_output = gr.Dataframe(
                            label="Top Matching Candidates",
                            headers=["ID", "Name", "Skills", "Experience", "Match %"],
                            datatype=["number", "str", "str", "str", "number"]
                        )
                        
                        match_plot = gr.Plot(label="Match Scores Visualization")
            
            with gr.Tab("💼 Find Jobs for Candidate"):
                with gr.Row():
                    with gr.Column():
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
                        
                        top_n_jobs = gr.Slider(
                            minimum=1, maximum=10, value=5,
                            label="Number of Top Jobs to Show"
                        )
                        
                        match_jobs_btn = gr.Button("Find Suitable Jobs", variant="primary")
                    
                    with gr.Column():
                        jobs_output = gr.Dataframe(
                            label="Top Matching Jobs",
                            headers=["ID", "Title", "Required Skills", "Level", "Match %"],
                            datatype=["number", "str", "str", "str", "number"]
                        )
                        
                        jobs_plot = gr.Plot(label="Job Match Visualization")
            
            with gr.Tab("📊 Skill Gap Analysis"):
                with gr.Row():
                    with gr.Column():
                        gap_job_select = gr.Dropdown(
                            choices=[f"{row['title']} (ID: {row['job_id']})" for _, row in self.jobs_df.iterrows()],
                            label="Select Job Position"
                        )
                        
                        gap_candidate_select = gr.Dropdown(
                            choices=[f"{row['name']} (ID: {row['candidate_id']})" for _, row in self.candidates_df.iterrows()],
                            label="Select Candidate"
                        )
                        
                        analyze_gap_btn = gr.Button("Analyze Skill Gap", variant="primary")
                    
                    with gr.Column():
                        gap_analysis_output = gr.JSON(label="Skill Gap Analysis")
                        
                        skills_radar = gr.Plot(label="Skills Radar Chart")
            
            # Event handlers for candidate matching
            job_dropdown.change(
                self.update_job_description,
                inputs=[job_dropdown],
                outputs=[job_description]
            )
            
            match_candidates_btn.click(
                self.find_matching_candidates,
                inputs=[job_dropdown, top_n_candidates],
                outputs=[candidates_output, match_plot]
            )
            
            # Event handlers for job matching
            candidate_dropdown.change(
                self.update_candidate_skills,
                inputs=[candidate_dropdown],
                outputs=[candidate_skills]
            )
            
            match_jobs_btn.click(
                self.find_matching_jobs,
                inputs=[candidate_dropdown, top_n_jobs],
                outputs=[jobs_output, jobs_plot]
            )
            
            # Event handler for skill gap analysis
            analyze_gap_btn.click(
                self.analyze_skill_gap,
                inputs=[gap_job_select, gap_candidate_select],
                outputs=[gap_analysis_output, skills_radar]
            )
            
            # Initialize with first values
            demo.load(
                self.initialize_values,
                outputs=[job_description, candidate_skills]
            )
        
        return demo
    
    def initialize_values(self):
        """Initialize form values"""
        first_job_desc = self.jobs_df.iloc[0]['description']
        first_candidate_skills = self.candidates_df.iloc[0]['skills']
        return first_job_desc, first_candidate_skills
    
    def update_job_description(self, job_selection):
        """Update job description based on selection"""
        job_id = int(job_selection.split("ID: ")[1].strip(")"))
        job_desc = self.jobs_df[self.jobs_df['job_id'] == job_id]['description'].iloc[0]
        return job_desc
    
    def update_candidate_skills(self, candidate_selection):
        """Update candidate skills based on selection"""
        candidate_id = int(candidate_selection.split("ID: ")[1].strip(")"))
        candidate_skills = self.candidates_df[self.candidates_df['candidate_id'] == candidate_id]['skills'].iloc[0]
        return candidate_skills
    
    def find_matching_candidates(self, job_selection, top_n):
        """Find matching candidates for a job"""
        job_id = int(job_selection.split("ID: ")[1].strip(")"))
        job_data = self.jobs_df[self.jobs_df['job_id'] == job_id].iloc[0]
        
        matched_candidates = self.skill_matcher.match_candidates_to_job(
            job_data['required_skills'],
            self.candidates_df,
            top_n
        )
        
        # Prepare output dataframe
        output_df = matched_candidates[['candidate_id', 'name', 'skills', 'experience', 'match_percentage']]
        
        # Create visualization
        fig = px.bar(
            output_df,
            x='name',
            y='match_percentage',
            title=f"Top {top_n} Candidate Matches for {job_data['title']}",
            labels={'name': 'Candidate', 'match_percentage': 'Match Percentage (%)'},
            color='match_percentage',
            color_continuous_scale='viridis'
        )
        fig.update_layout(xaxis_tickangle=-45)
        
        return output_df, fig
    
    def find_matching_jobs(self, candidate_selection, top_n):
        """Find matching jobs for a candidate"""
        candidate_id = int(candidate_selection.split("ID: ")[1].strip(")"))
        candidate_data = self.candidates_df[self.candidates_df['candidate_id'] == candidate_id].iloc[0]
        
        matched_jobs = self.skill_matcher.find_jobs_for_candidate(
            candidate_data['skills'],
            self.jobs_df,
            top_n
        )
        
        # Prepare output dataframe
        output_df = matched_jobs[['job_id', 'title', 'required_skills', 'experience_level', 'match_percentage']]
        
        # Create visualization
        fig = px.bar(
            output_df,
            x='title',
            y='match_percentage',
            title=f"Top {top_n} Job Matches for {candidate_data['name']}",
            labels={'title': 'Job Title', 'match_percentage': 'Match Percentage (%)'},
            color='match_percentage',
            color_continuous_scale='plasma'
        )
        fig.update_layout(xaxis_tickangle=-45)
        
        return output_df, fig
    
    def analyze_skill_gap(self, job_selection, candidate_selection):
        """Analyze skill gap between job and candidate"""
        if not job_selection or not candidate_selection:
            return {}, self.create_empty_plot()
        
        job_id = int(job_selection.split("ID: ")[1].strip(")"))
        candidate_id = int(candidate_selection.split("ID: ")[1].strip(")"))
        
        job_data = self.jobs_df[self.jobs_df['job_id'] == job_id].iloc[0]
        candidate_data = self.candidates_df[self.candidates_df['candidate_id'] == candidate_id].iloc[0]
        
        gap_analysis = self.skill_matcher.skill_gap_analysis(
            job_data['required_skills'],
            candidate_data['skills']
        )
        
        # Create radar chart
        radar_fig = self.create_skills_radar_chart(gap_analysis, job_data['title'], candidate_data['name'])
        
        return gap_analysis, radar_fig
    
    def create_skills_radar_chart(self, gap_analysis, job_title, candidate_name):
        """Create a radar chart for skills comparison"""
        categories = ['Skill Match', 'Coverage', 'Alignment']
        
        # Calculate metrics for radar chart
        match_score = gap_analysis['match_percentage'] / 100
        coverage_score = min(1.0, len(gap_analysis['matching_skills']) / 10)  # Normalize
        alignment_score = match_score * 0.7 + coverage_score * 0.3  # Weighted score
        
        values = [match_score, coverage_score, alignment_score]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=f'{candidate_name} Skills',
            line_color='blue'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=f"Skills Radar: {candidate_name} vs {job_title}"
        )
        
        return fig
    
    def create_empty_plot(self):
        """Create an empty plot placeholder"""
        fig = go.Figure()
        fig.update_layout(
            title="Select both job and candidate to see analysis",
            xaxis_title="",
            yaxis_title="",
            annotations=[dict(
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                text="No data to display",
                showarrow=False,
                font=dict(size=16)
            )]
        )
        return fig

def main():
    """Main function to run the application"""
    app = SkillMatcherApp()
    demo = app.create_interface()
    demo.launch(share=True, debug=True)

if __name__ == "__main__":
    main()