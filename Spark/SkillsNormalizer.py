import os
import json

from pyspark import SparkFiles

class SkillNormalizer:
    
    normalization_map = None

    @classmethod
    def load_normalization_map(cls):
        """ Load normalization map from config directory """
        
        if cls.normalization_map is None:
            config_path = SparkFiles.get('normalization_map.json')
            try:
                with open(config_path, 'r') as f:
                    cls.normalization_map = json.load(f)
            except Exception as e:
                print(f'Error loading the normalization dictionary: {e}')
                cls.normalization_map = {}

    @classmethod
    def normalize_skill(cls, skill):
        """ Normalize a single skill name """

        if not skill:
            return skill
        if skill in cls.normalization_map:
            return cls.normalization_map[skill]   
        return skill
    
    @classmethod
    def normalize_skills_list(cls, raw_skills):
        """ Normalize a list of skills """

        if not raw_skills:
            return []
        
        cls.load_normalization_map()

        normalized_skills = set()
        for skill in raw_skills:
            normalized = cls.normalize_skill(skill)
            if normalized:
                normalized_skills.add(normalized)          
        return normalized_skills