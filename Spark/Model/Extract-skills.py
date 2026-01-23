import json
import torch
from tqdm import tqdm
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForTokenClassification


class JobSkillsExtractor:
    """Extract skills from job descriptions using fine-tuned model"""
    
    def __init__(self, model_path: str = "./skill_extractor_model"):
        """ Initialize the skill extractor """

        print(f"Loading model from {model_path}...")
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        
        # Get label mappings from model config
        self.id2label = self.model.config.id2label
    

    def _reconstruct_word(self, tokens: List[str]) -> str:
        if not tokens:
            return ""
        return self.tokenizer.convert_tokens_to_string(tokens).strip()


    def extract_skills(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        
        # Tokenize - use max_length to handle long descriptions
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=False
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
        
        # Get tokens and labels
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_labels = [self.id2label[p.item()] for p in predictions[0]]
        
        # Extract skills using BIO tagging
        skills = set()
        current_skill = []
        
        for token, label in zip(tokens, predicted_labels):
            # Skip special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            if label == "B-SKILL":
                # Save previous skill if exists
                if current_skill:
                    skill_text = self._reconstruct_word(current_skill)
                    if skill_text:
                        skills.add(skill_text)
                # Start new skill
                current_skill = [token]
                
            elif label == "I-SKILL" and current_skill:
                # Continue current skill
                current_skill.append(token)
                
            else:
                # End of skill (O tag)
                if current_skill:
                    skill_text = self._reconstruct_word(current_skill)
                    if skill_text:
                        skills.add(skill_text)
                    current_skill = []
        
        # Last skill
        if current_skill:
            skill_text = self._reconstruct_word(current_skill)
            if skill_text:
                skills.add(skill_text)
        
        return list(skills)

    
    def process_job_post(self, job_post: Dict) -> Dict:
        result = job_post.copy()
        
        # Extract skills from description
        description = job_post.get("Description", "")
        skills = self.extract_skills(description)
        
        # Remove description and add skills
        result.pop("Description", None)
        result["Skills"] = skills
        return result
    

    def process_json_file(self, input_file: str, output_file: str, show_progress: bool = True):
        """ Process a JSON file with multiple job posts """

        print(f"Reading job posts from {input_file}...")
        
        # Load input file
        with open(input_file, 'r', encoding='utf-8') as f:
            job_posts = json.load(f)
        
        # Handle both single object and array of objects
        if isinstance(job_posts, dict):
            job_posts = [job_posts]
        
        print(f"Found {len(job_posts)} job posts")
        
        # Process each job post
        results = []
        iterator = tqdm(job_posts) if show_progress else job_posts
        
        for job_post in iterator:
            try:
                result = self.process_job_post(job_post)
                results.append(result)
            except Exception as e:
                print(f"Error processing job {job_post.get('Job_ID', 'unknown')}: {e}")
                # Keep original job post on error
                results.append(job_post)
        
        # Save results
        print(f"\nSaving results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Successfully processed {len(results)} job posts!")


def main():
    print("="*60)
    print("JOB SKILLS EXTRACTOR")
    print("="*60 + "\n")
    
    # Initialize extractor
    extractor = JobSkillsExtractor(model_path="./skill_extractor_model")
    
    # Process JSON file
    input_file = "LinkedinJobPosts.json"
    output_file = "Skills.json"
    
    extractor.process_json_file(
        input_file=input_file,
        output_file=output_file,
        show_progress=True
    )


if __name__ == "__main__":
    main()