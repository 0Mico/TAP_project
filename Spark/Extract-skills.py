import json
from pathlib import Path
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm


class JobSkillsExtractor:
    """Extract skills from job descriptions using fine-tuned model"""
    
    def __init__(self, model_path: str = "./skill_extractor_model"):
        """
        Initialize the skill extractor
        
        Args:
            model_path: Path to the fine-tuned model directory
        """
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        
        # Get label mappings from model config
        self.id2label = self.model.config.id2label
        print(f"✓ Model loaded successfully!")
        print(f"Labels: {list(self.id2label.values())}\n")
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract skills from a text description
        
        Args:
            text: Job description text
            
        Returns:
            List of extracted skills
        """
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
        
        # Don't forget last skill
        if current_skill:
            skill_text = self._reconstruct_word(current_skill)
            if skill_text:
                skills.add(skill_text)
        
        return list(skills)
    
    def _reconstruct_word(self, tokens: List[str]) -> str:
        """
        Reconstruct a word from BERT tokens
        
        Args:
            tokens: List of BERT tokens (may include ## for subwords)
            
        Returns:
            Reconstructed word/phrase
        """
        if not tokens:
            return ""
               
        return self.tokenizer.convert_tokens_to_string(tokens).strip()

    
    def process_job_post(self, job_post: Dict) -> Dict:
        """
        Process a single job post and extract skills
        
        Args:
            job_post: Dictionary with job post data
            
        Returns:
            New dictionary with skills instead of description
        """
        # Create a copy of the job post
        result = job_post.copy()
        
        # Extract skills from description
        description = job_post.get("Description", "")
        skills = self.extract_skills(description)
        
        # Remove description and add skills
        result.pop("Description", None)
        result["Skills"] = skills
        
        return result
    
    def process_json_file(
        self, 
        input_file: str, 
        output_file: str,
        show_progress: bool = True
        ):
        """
        Process an entire JSON file with multiple job posts
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output JSON file
            show_progress: Whether to show progress bar
        """
        print(f"Reading job posts from {input_file}...")
        
        # Load input file
        with open(input_file, 'r', encoding='utf-8') as f:
            job_posts = json.load(f)
        
        # Handle both single object and array of objects
        if isinstance(job_posts, dict):
            job_posts = [job_posts]
        
        print(f"Found {len(job_posts)} job posts")
        print("Extracting skills...\n")
        
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
        
        # Show some statistics
        total_skills = sum(len(r.get("Skills", [])) for r in results)
        avg_skills = total_skills / len(results) if results else 0
        print(f"\nStatistics:")
        print(f"  Total skills extracted: {total_skills}")
        print(f"  Average skills per job: {avg_skills:.1f}")
    
    def process_single_job(self, job_post_dict: Dict) -> Dict:
        """
        Process a single job post dictionary
        
        Args:
            job_post_dict: Job post dictionary
            
        Returns:
            Processed dictionary with skills
        """
        return self.process_job_post(job_post_dict)


def main():
    """Example usage"""
    print("="*60)
    print("JOB SKILLS EXTRACTOR")
    print("="*60 + "\n")
    
    # Initialize extractor
    extractor = JobSkillsExtractor(model_path="./skill_extractor_model_942")
    
    # Process JSON file
    input_file = "Test-extraction.json"  # Change to your input file
    output_file = "Skills.json"  # Change to your output file
    
    extractor.process_json_file(
        input_file=input_file,
        output_file=output_file,
        show_progress=True
    )
    
    print("\n" + "="*60)
    print("EXAMPLE OUTPUT")
    print("="*60)
    
    # Show example of first job
    with open(output_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
        if results:
            print(json.dumps(results[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()