import json

from pyspark import SparkFiles

class JobTitleNormalizer:
    
    normalization_map = None

    @classmethod
    def load_normalization_map(cls):
        """ Load normalization map from config directory """
        
        if cls.normalization_map is None:
            config_path = SparkFiles.get('job_titles_normalization_map.json')
            try:
                with open(config_path, 'r') as f:
                    cls.normalization_map = json.load(f)
            except Exception as e:
                print(f'Error loading the normalization dictionary: {e}')
                cls.normalization_map = {}

    @classmethod
    def normalize_job_title(cls, title):
        """ Normalize job title of the current job post """

        if cls.normalization_map is None:
            cls.load_normalization_map()
            
        if not title:
            return title
        if title in cls.normalization_map:
            return cls.normalization_map[title]   
        return title