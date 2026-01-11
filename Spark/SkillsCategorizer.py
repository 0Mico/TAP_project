import os
import json

from pyspark import SparkFiles

class SkillCategorizer:
    """ Categorizes normalized skills into technology categories """

    cloud_services_division = None
    technologies = None

    @classmethod
    def load_configs(cls):
        """ Loads skill->provider and skill->topic association """

        if cls.cloud_services_division is None: 
            cloud_config = SparkFiles.get('cloud_services.json')
            with open(cloud_config) as f:
                data = json.load(f)
                cls.cloud_services_division = {
                    provider: set(services) for provider, services in data.items()
                }
        if cls.technologies is None:
            technologies_config = SparkFiles.get('technologies.json')
            with open(technologies_config) as f:
                data = json.load(f)
                cls.technologies = {
                    topic: set(technology) for topic, technology in data.items()
                }  

    @classmethod
    def _get_empty_categories(cls):        
        if cls.technologies is None:
            cls.load_configs()
        return {topic: set() for topic in cls.technologies.keys()}
    
    @classmethod
    def _get_empty_cloud_services(cls):
        if cls.cloud_services_division is None:
            cls.load_configs()
        return {provider: set() for provider in cls.cloud_services_division.keys()}
       
    @classmethod
    def categorize_skills(cls, normalized_skills):
        """ Divide normalized skills in provider and topic categorization """

        if not normalized_skills:
            return {
                'raw_skills': [],
                'categories': cls._get_empty_categories(),
                'cloud_services': cls._get_empty_cloud_services()
            }

        cls.load_configs()
        categories = cls._get_empty_categories()
        cloud_services = cls._get_empty_cloud_services()
        print(categories)
        print(cloud_services)
        
        for skill in normalized_skills:
            skill_lower = skill.lower()

            for provider, services in cls.cloud_services_division.items():
                if skill_lower in services:
                    categories['cloud_providers'].add(provider)
                    cloud_services[str(provider)].add(skill_lower)

            for topic, technologies in cls.technologies.items():
                if skill_lower in technologies:
                    categories[topic].add(skill_lower)

        return {
            'raw_skills': sorted(list(normalized_skills)),
            'categories': {k: sorted(list(v)) for k, v in categories.items()},
            'cloud_services': {k: sorted(list(v)) for k, v in cloud_services.items()}
        }