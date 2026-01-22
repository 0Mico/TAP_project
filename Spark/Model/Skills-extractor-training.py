import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from sklearn.model_selection import train_test_split
import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


class SkillNERDataset(Dataset):
    """Custom Dataset for NER training"""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)


class DatasetPreparator:
    """Prepare dataset from labeled JSON files"""
    
    def __init__(self, json_dir: str = "./job_posts"):
        self.json_dir = json_dir
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
        
        # Label mapping
        self.label2id = {
            "O": 0,
            "B-SKILL": 1,
            "I-SKILL": 2
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
    def load_json_files(self) -> List[Dict]:
        """Load all JSON files from directory"""
        json_files = list(Path(self.json_dir).glob("*.json"))
        
        print(f"Found {len(json_files)} JSON files")
        
        job_posts = []
        invalid_count = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    job_data = json.load(f)
                    
                    # Validate structure
                    if 'tokens' in job_data and job_data['tokens']:
                        job_posts.append(job_data)
                    else:
                        invalid_count += 1
                        print(f"⚠️  Skipping {json_file.name}: No tokens found")
                        
            except json.JSONDecodeError as e:
                invalid_count += 1
                print(f"⚠️  Skipping {json_file.name}: Invalid JSON - {e}")
            except Exception as e:
                invalid_count += 1
                print(f"⚠️  Skipping {json_file.name}: {e}")
        
        print(f"✓ Successfully loaded: {len(job_posts)} jobs")
        if invalid_count > 0:
            print(f"✗ Invalid files: {invalid_count}")
        
        return job_posts
    
    def extract_tokens_and_labels(self, job_posts: List[Dict]) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Extract tokens and labels from job posts
        
        Returns:
            Tuple of (tokens_list, labels_list)
        """
        tokens_list = []
        labels_list = []
        
        for job in job_posts:
            tokens = []
            labels = []
            
            # Sort tokens by position to ensure correct order
            sorted_tokens = sorted(job['tokens'], key=lambda x: x['position'])
            
            for token_obj in sorted_tokens:
                # Extract text and label
                text = token_obj['text']
                label = token_obj['label']
                
                tokens.append(text)
                labels.append(label)
            
            if tokens and labels:
                tokens_list.append(tokens)
                labels_list.append(labels)
        
        return tokens_list, labels_list
    
    def show_dataset_statistics(self, tokens_list: List[List[str]], labels_list: List[List[str]]):
        """Display statistics about the dataset"""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        total_jobs = len(tokens_list)
        total_tokens = sum(len(tokens) for tokens in tokens_list)
        
        # Count skills
        skill_counts = {
            'B-SKILL': 0,
            'I-SKILL': 0,
            'O': 0
        }
        
        for labels in labels_list:
            for label in labels:
                skill_counts[label] = skill_counts.get(label, 0) + 1
        
        # Count unique skills (number of B-SKILL tags)
        total_skills = skill_counts['B-SKILL']
        
        # Calculate averages
        avg_tokens = total_tokens / total_jobs if total_jobs > 0 else 0
        avg_skills = total_skills / total_jobs if total_jobs > 0 else 0
        
        print(f"\nLabel Distribution:")
        print(f"  O (Outside):        {skill_counts['O']:,} ({skill_counts['O']/total_tokens*100:.1f}%)")
        print(f"  B-SKILL (Begin):    {skill_counts['B-SKILL']:,} ({skill_counts['B-SKILL']/total_tokens*100:.1f}%)")
        print(f"  I-SKILL (Inside):   {skill_counts['I-SKILL']:,} ({skill_counts['I-SKILL']/total_tokens*100:.1f}%)")
        
        print("="*60 + "\n")
    
    def prepare_encodings(self, tokens_list: List[List[str]], labels_list: List[List[str]]) -> Tuple[Dict, List[List[int]]]:
        """
        Prepare encodings compatible with BERT tokenizer
        
        The key challenge: Your tokens are already pre-tokenized by BERT,
        so we need to align them correctly with BERT's tokenization
        """
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        
        for tokens, labels in zip(tokens_list, labels_list):
            # Convert tokens to input IDs
            # Use convert_tokens_to_ids since tokens are already BERT tokens
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            # Add [CLS] at start and [SEP] at end
            input_ids = [self.tokenizer.cls_token_id] + token_ids + [self.tokenizer.sep_token_id]
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * len(input_ids)
            
            # Convert labels to IDs and add special token labels
            label_ids = [self.label2id[label] for label in labels]
            # -100 is the ignore index for loss calculation (for [CLS] and [SEP])
            label_ids = [-100] + label_ids + [-100]
            
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(label_ids)
        
        return {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_masks
        }, all_labels
    
    def pad_sequences(self, encodings: Dict, labels: List[List[int]]) -> Tuple[Dict, List[List[int]]]:
        """Pad sequences to the same length"""
        max_length = max(len(seq) for seq in encodings['input_ids'])
        
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        
        for input_ids, attention_mask, label_ids in zip(
            encodings['input_ids'], 
            encodings['attention_mask'], 
            labels
        ):
            padding_length = max_length - len(input_ids)
            
            # Pad input_ids with tokenizer's pad_token_id
            padded_input_ids.append(
                input_ids + [self.tokenizer.pad_token_id] * padding_length
            )
            
            # Pad attention_mask with 0s
            padded_attention_masks.append(
                attention_mask + [0] * padding_length
            )
            
            # Pad labels with -100 (ignore index)
            padded_labels.append(
                label_ids + [-100] * padding_length
            )
        
        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_masks
        }, padded_labels
    
    def create_datasets(self, test_size: float = 0.2, random_seed: int = 5) -> Tuple[SkillNERDataset, SkillNERDataset]:
        """
        Create train and validation datasets
        
        Args:
            test_size: Proportion for validation set (default 0.2 = 20%)
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        print("Loading and preparing dataset...\n")
        
        # Load JSON files
        job_posts = self.load_json_files()
        
        if not job_posts:
            raise ValueError("No valid job posts found!")
        
        # Extract tokens and labels
        tokens_list, labels_list = self.extract_tokens_and_labels(job_posts)
        
        # Show statistics
        self.show_dataset_statistics(tokens_list, labels_list)
        
        # Split into train and validation
        train_tokens, val_tokens, train_labels, val_labels = train_test_split(
            tokens_list, labels_list, 
            test_size=test_size, 
            random_state=random_seed
        )
        
        print(f"Dataset Split:")
        print(f"  Training samples:   {len(train_tokens)} ({(1-test_size)*100:.0f}%)")
        print(f"  Validation samples: {len(val_tokens)} ({test_size*100:.0f}%)\n")
        
        # Prepare encodings
        print("Encoding training data...")
        train_encodings, train_label_ids = self.prepare_encodings(train_tokens, train_labels)
        
        print("Encoding validation data...")
        val_encodings, val_label_ids = self.prepare_encodings(val_tokens, val_labels)
        
        # Pad sequences
        print("Padding sequences...")
        train_encodings, train_label_ids = self.pad_sequences(train_encodings, train_label_ids)
        val_encodings, val_label_ids = self.pad_sequences(val_encodings, val_label_ids)
        
        # Create datasets
        train_dataset = SkillNERDataset(train_encodings, train_label_ids)
        val_dataset = SkillNERDataset(val_encodings, val_label_ids)
        
        print("✓ Datasets created successfully!\n")
        
        return train_dataset, val_dataset


class SkillExtractorTrainer:
    """Trainer for skill extraction model"""
    
    def __init__(self, model_name: str = "bert-base-multilingual-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.label2id = {"O": 0, "B-SKILL": 1, "I-SKILL": 2}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        self.model = None
    
    def compute_metrics(self, pred):
        """Compute evaluation metrics"""
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_labels = [
            [self.id2label[l] for l in label if l != -100] 
            for label in labels
        ]
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        return {
            'precision': precision_score(true_labels, true_predictions),
            'recall': recall_score(true_labels, true_predictions),
            'f1': f1_score(true_labels, true_predictions),
        }
    
    def train(
        self, 
        train_dataset: SkillNERDataset, 
        val_dataset: SkillNERDataset,
        output_dir: str = "./skill_extractor_model",
        num_epochs: int = 15,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1
    ):
        """Train the model"""
        
        print("="*60)
        print("TRAINING CONFIGURATION")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        #print(f"Warmup steps: {warmup_steps}")
        print(f"Warmup ratio: {warmup_ratio}")
        print("="*60 + "\n")
        
        # Initialize model
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_steps=50,
            #warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            push_to_hub=False,
            label_smoothing_factor=0.1,
            max_grad_norm=1.0
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
        )
        
        # Train
        print("Starting training...\n")
        trainer.train()
        
        # Save model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"\n✓ Model saved to {output_dir}")
        
        # Final evaluation
        print("\nFinal Evaluation:")
        results = trainer.evaluate()
        print(f"  Precision: {results['eval_precision']:.4f}")
        print(f"  Recall:    {results['eval_recall']:.4f}")
        print(f"  F1 Score:  {results['eval_f1']:.4f}")
        
        return trainer


def main():
    print("\n" + "="*60)
    print("SKILL EXTRACTION MODEL - TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # Step 1: Prepare dataset
    preparator = DatasetPreparator(json_dir="./job-posts")
    train_dataset, val_dataset = preparator.create_datasets(test_size=0.2, random_seed=15)
    
    # Step 2: Train model
    trainer = SkillExtractorTrainer()
    trained_model = trainer.train(
        train_dataset,
        val_dataset,
        output_dir="./skill_extractor_model",
        num_epochs=15,        # Good for small dataset (200 samples)
        batch_size=8,        # Adjust based on your GPU memory
        learning_rate=3e-5,  # Conservative for fine-tuning
        warmup_ratio=0.1
    )

if __name__ == "__main__":
    main()