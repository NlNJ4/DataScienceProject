
import sys
import os
import argparse

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pipelines.scraper import Scraper
    from pipelines.cleaner import Cleaner
    from pipelines.feature_engineer import FeatureEngineer
    from pipelines.modeler import Modeler
    from pipelines.visualizer import Visualizer
except ImportError:
    from scraper import Scraper
    from cleaner import Cleaner
    from feature_engineer import FeatureEngineer
    from modeler import Modeler
    from visualizer import Visualizer

class DataPipeline:
    def __init__(self):
        pass
    
    def run_scraper(self):
        print("\n=== STEP 1: SCRAPING DATA ===")
        scraper = Scraper()
        scraper.run_all()

    def run_cleaner(self):
        print("\n=== STEP 2: CLEANING DATA ===")
        cleaner = Cleaner()
        cleaner.run()

    def run_feature_engineering(self):
        print("\n=== STEP 3: FEATURE ENGINEERING ===")
        fe = FeatureEngineer()
        fe.run()

    def run_modeling(self):
        print("\n=== STEP 4: MODEL TRAINING ===")
        modeler = Modeler()
        modeler.run()

    def run_visualization(self):
        print("\n=== STEP 5: VISUALIZATION ===")
        viz = Visualizer()
        viz.generate_report()

    def run_all(self):
        self.run_scraper()
        self.run_cleaner()
        self.run_feature_engineering()
        self.run_modeling()
        self.run_visualization()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffy Fondue Data Pipeline")
    parser.add_argument('--step', type=str, choices=['scrape', 'clean', 'feature', 'model', 'visualize', 'all'], 
                        default='all', help='Pipeline step to run')
    
    args = parser.parse_args()
    
    pipeline = DataPipeline()
    
    if args.step == 'scrape':
        pipeline.run_scraper()
    elif args.step == 'clean':
        pipeline.run_cleaner()
    elif args.step == 'feature':
        pipeline.run_feature_engineering()
    elif args.step == 'model':
        pipeline.run_modeling()
    elif args.step == 'visualize':
        pipeline.run_visualization()
    elif args.step == 'all':
        pipeline.run_all()
