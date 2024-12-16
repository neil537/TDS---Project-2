#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import json
import base64
from io import BytesIO, StringIO
import requests
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import chardet
except ImportError:
    print("Please install chardet: pip install chardet")
    sys.exit(1)

class DatasetAnalyzer:
    """A class to analyze different types of datasets with specialized analysis methods."""
    
    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise ValueError(f"CSV file not found: {csv_path}")
            
        self.csv_path = csv_path
        # Try reading with different encoding configurations
        try:
            # First attempt: UTF-8 with error handling
            self.df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='replace')
        except Exception:

            # Second attempt: Let Python detect the encoding
            import chardet
            with open(csv_path, 'rb') as file:
                raw_data = file.read()
            detected = chardet.detect(raw_data)
            self.df = pd.read_csv(csv_path, encoding=detected['encoding'])
        self.token = os.environ.get("AIPROXY_TOKEN")
        if not self.token:
            raise ValueError("AIPROXY_TOKEN environment variable not set")
        
        # Create output directory based on CSV name
        self.output_dir = os.path.splitext(os.path.basename(csv_path))[0]
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set visualization parameters
        self.fig_size = (10, 6)
        self.dpi = 100
        
        # Determine dataset type
        self.dataset_type = self._determine_dataset_type()

    def _determine_dataset_type(self) -> str:
        """Determine the type of dataset based on column names and content."""
        columns = set(self.df.columns.str.lower())
        
        # Check for book-related columns
        if any(col in columns for col in ['book_id', 'title', 'authors', 'isbn']):
            return 'books'
        
        # Check for happiness/country-related columns
        if any(col in columns for col in ['country', 'happiness', 'score', 'gdp']):
            return 'happiness'
        
        # Check for media/ratings columns
        if any(col in columns for col in ['rating', 'genre', 'release']):
            return 'media'
            
        return 'generic'

    def create_visualizations(self) -> List[str]:
        """Create and save visualizations based on dataset type."""
        image_paths = []
        print(f"Creating visualizations for {self.dataset_type} dataset")
        
        try:
            plt.style.use('seaborn-v0_8')
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Common visualizations for all dataset types
            corr_path = self._create_correlation_heatmap()
            if corr_path:
                image_paths.append(corr_path)
                
            # Create missing values visualization for all types
            missing_path = self._create_missing_values_chart()
            if missing_path:
                image_paths.append(missing_path)
                
            if self.dataset_type == 'books':
                # Books specific visualizations
                rating_dist_path = self._create_rating_distribution()
                if rating_dist_path:
                    image_paths.append(rating_dist_path)
                    
                year_dist_path = self._create_year_distribution()
                if year_dist_path:
                    image_paths.append(year_dist_path)
                    
                authors_path = self._create_top_authors()
                if authors_path:
                    image_paths.append(authors_path)
                    
            elif self.dataset_type == 'happiness':
                # Happiness specific visualizations
                score_dist_path = self._create_happiness_distribution()
                if score_dist_path:
                    image_paths.append(score_dist_path)
                    
                countries_path = self._create_top_countries()
                if countries_path:
                    image_paths.append(countries_path)
                    
                factors_path = self._create_happiness_factors()
                if factors_path:
                    image_paths.append(factors_path)
                    
            elif self.dataset_type == 'media':
                # Media specific visualizations
                rating_path = self._create_media_rating_distribution()
                if rating_path:
                    image_paths.append(rating_path)
                    
                genre_path = self._create_genre_distribution()
                if genre_path:
                    image_paths.append(genre_path)
                    
                release_path = self._create_release_trend()
                if release_path:
                    image_paths.append(release_path)
                    
            else:  # generic dataset
                # Ensure at least 3 visualizations for generic datasets
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                for i, col in enumerate(numeric_cols[:3]):
                    dist_path = self._create_generic_distribution(col)
                    if dist_path:
                        image_paths.append(dist_path)
                        
            # If we still don't have 3 visualizations, add some general ones
            while len(image_paths) < 3:
                if len(image_paths) == 1:
                    path = self._create_data_overview()
                elif len(image_paths) == 2:
                    path = self._create_column_types_chart()
                else:
                    path = self._create_value_counts_chart()
                    
                if path:
                    image_paths.append(path)
            
            print(f"Created {len(image_paths)} visualizations")
            return image_paths[:3]
            
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            return image_paths

    def _create_rating_distribution(self) -> str:
        """Create rating distribution visualization."""
        if 'average_rating' not in self.df.columns:
            return None
            
        plt.figure(figsize=self.fig_size)
        sns.histplot(data=self.df['average_rating'].dropna(), bins=20)
        plt.title('Distribution of Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        
        filepath = os.path.join(self.output_dir, 'rating_distribution.png')
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return filepath

    def _create_year_distribution(self) -> str:
        """Create publication year distribution visualization."""
        year_col = next((col for col in self.df.columns if 'year' in col.lower()), None)
        if not year_col:
            return None
            
        plt.figure(figsize=self.fig_size)
        sns.histplot(data=self.df[year_col].dropna(), bins=30)
        plt.title('Distribution of Publication Years')
        plt.xlabel('Year')
        plt.ylabel('Count')
        
        filepath = os.path.join(self.output_dir, 'year_distribution.png')
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return filepath

    def _create_top_authors(self) -> str:
        """Create top authors visualization."""
        if 'authors' not in self.df.columns:
            return None
            
        top_authors = self.df['authors'].value_counts().head(10)
        
        plt.figure(figsize=self.fig_size)
        sns.barplot(x=top_authors.values, y=top_authors.index)
        plt.title('Top 10 Authors by Number of Books')
        plt.xlabel('Number of Books')
        
        filepath = os.path.join(self.output_dir, 'top_authors.png')
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return filepath

    def _create_happiness_distribution(self) -> str:
        """Create happiness score distribution visualization."""
        happiness_cols = [col for col in self.df.columns if 'happiness' in col.lower() or 'score' in col.lower()]
        if not happiness_cols:
            return None
            
        plt.figure(figsize=self.fig_size)
        sns.histplot(data=self.df[happiness_cols[0]].dropna(), bins=30)
        plt.title(f'Distribution of {happiness_cols[0]}')
        plt.xlabel(happiness_cols[0])
        plt.ylabel('Count')
        
        filepath = os.path.join(self.output_dir, 'happiness_distribution.png')
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return filepath

    def _create_top_countries(self) -> str:
        """Create top countries visualization."""
        if 'country' not in self.df.columns or not any('score' in col.lower() or 'happiness' in col.lower() for col in self.df.columns):
            return None
            
        score_col = next(col for col in self.df.columns if 'score' in col.lower() or 'happiness' in col.lower())
        top_countries = self.df.nlargest(15, score_col)[['country', score_col]]
        
        plt.figure(figsize=self.fig_size)
        sns.barplot(data=top_countries, x=score_col, y='country')
        plt.title(f'Top 15 Countries by {score_col}')
        
        filepath = os.path.join(self.output_dir, 'top_countries.png')
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return filepath

    def _create_media_rating_distribution(self) -> str:
        """Create media rating distribution visualization."""
        rating_cols = [col for col in self.df.columns if 'rating' in col.lower()]
        if not rating_cols:
            return None
            
        plt.figure(figsize=self.fig_size)
        sns.histplot(data=self.df[rating_cols[0]].dropna(), bins=20)
        plt.title(f'Distribution of {rating_cols[0]}')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        
        filepath = os.path.join(self.output_dir, 'media_rating_distribution.png')
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return filepath

    def _create_genre_distribution(self) -> str:
        """Create genre distribution visualization."""
        if 'genre' not in self.df.columns:
            return None
            
        top_genres = self.df['genre'].value_counts().head(10)
        
        plt.figure(figsize=self.fig_size)
        sns.barplot(x=top_genres.values, y=top_genres.index)
        plt.title('Top 10 Genres')
        plt.xlabel('Count')
        
        filepath = os.path.join(self.output_dir, 'genre_distribution.png')
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return filepath

    def _create_release_trend(self) -> str:
        """Create release trend visualization."""
        release_cols = [col for col in self.df.columns if 'release' in col.lower() or 'year' in col.lower()]
        if not release_cols:
            return None
            
        plt.figure(figsize=self.fig_size)
        year_data = self.df[release_cols[0]].dropna()
        sns.histplot(data=year_data, bins=30)
        plt.title('Release Year Distribution')
        plt.xlabel('Year')
        plt.ylabel('Count')
        
        filepath = os.path.join(self.output_dir, 'release_trend.png')
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return filepath

    def _create_missing_values_chart(self) -> str:
        """Create visualization of missing values."""
        missing_data = (self.df.isnull().sum() / len(self.df) * 100).sort_values(ascending=True)
        
        plt.figure(figsize=self.fig_size)
        missing_data.plot(kind='barh')
        plt.title('Percentage of Missing Values by Column')
        plt.xlabel('Percentage Missing')
        
        filepath = os.path.join(self.output_dir, 'missing_values.png')
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return filepath

    def _create_happiness_factors(self) -> str:
        """Create visualization of factors affecting happiness."""
        if not any(col.lower().endswith('score') for col in self.df.columns):
            return None
            
        # Find all score columns except the main happiness score
        factor_cols = [col for col in self.df.columns if col.lower().endswith('score') 
                      and 'happiness' not in col.lower()]
        
        if not factor_cols:
            return None
            
        plt.figure(figsize=self.fig_size)
        self.df[factor_cols].mean().sort_values().plot(kind='barh')
        plt.title('Average Factor Scores')
        plt.xlabel('Score')
        
        filepath = os.path.join(self.output_dir, 'happiness_factors.png')
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return filepath

    def _create_data_overview(self) -> str:
        """Create a general overview of the dataset."""
        plt.figure(figsize=self.fig_size)
        
        # Get counts of different data types
        dtype_counts = self.df.dtypes.value_counts()
        plt.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        plt.title('Distribution of Column Types')
        
        filepath = os.path.join(self.output_dir, 'data_overview.png')
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return filepath

    def _create_column_types_chart(self) -> str:
        """Create a visualization of column data types."""
        type_counts = self.df.dtypes.astype(str).value_counts()
        
        plt.figure(figsize=self.fig_size)
        sns.barplot(x=type_counts.values, y=type_counts.index)
        plt.title('Column Data Types')
        plt.xlabel('Count')
        
        filepath = os.path.join(self.output_dir, 'column_types.png')
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return filepath

    def _create_value_counts_chart(self) -> str:
        """Create a visualization of unique value counts for each column."""
        unique_counts = self.df.nunique().sort_values(ascending=True)
        
        plt.figure(figsize=self.fig_size)
        unique_counts.plot(kind='barh')
        plt.title('Number of Unique Values per Column')
        plt.xlabel('Count of Unique Values')
        
        filepath = os.path.join(self.output_dir, 'unique_values.png')
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return filepath

    def get_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive data summary."""
        summary = {
            "dataset_type": self.dataset_type,
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
        }
        
        return summary

    def generate_narrative(self, analysis_results: Dict[str, Any], image_paths: List[str]) -> str:
        """Generate analysis narrative."""
        narrative = f"""# Dataset Analysis Report

## Overview
- Dataset Type: {analysis_results['dataset_type']}
- Number of Records: {analysis_results['shape'][0]}
- Number of Features: {analysis_results['shape'][1]}

## Key Features
{', '.join(analysis_results['columns'])}

## Data Quality
Missing values per column:
"""
        
        for col, missing in analysis_results['missing_values'].items():
            if missing > 0:
                narrative += f"- {col}: {missing}\n"
        
        if image_paths:
            narrative += "\n## Visualizations\n"
            for path in image_paths:
                name = os.path.basename(path)
                narrative += f"\n![{name}]({name})\n"
        
        # Save narrative
        with open(os.path.join(self.output_dir, 'README.md'), 'w') as f:
            f.write(narrative)
        
        return narrative

def main():
    """Main function to run the analysis."""
    if len(sys.argv) != 2:
        script_name = os.path.basename(sys.argv[0])
        print(f"Usage: python3 {script_name} <csv_file>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    try:
        # Initialize analyzer
        analyzer = DatasetAnalyzer(csv_path)
        print(f"Analyzing {analyzer.dataset_type} dataset...")
        
        # Perform analysis
        analysis_results = analyzer.get_data_summary()
        print("Analysis completed...")
        
        # Generate visualizations
        image_paths = analyzer.create_visualizations()
        print(f"Created {len(image_paths)} visualizations...")
        
        # Generate narrative
        analyzer.generate_narrative(analysis_results, image_paths)
        print(f"Analysis complete. Results saved in {analyzer.output_dir}/")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()