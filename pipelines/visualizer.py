
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
try:
    from pipelines.config import TRAFFY_MERGED_PATH
except ImportError:
    from config import TRAFFY_MERGED_PATH

class Visualizer:
    def __init__(self):
        pass

    def load_data(self):
        print(f"Loading data from {TRAFFY_MERGED_PATH}...")
        try:
            df = pd.read_csv(TRAFFY_MERGED_PATH)
            # Ensure timestamps are datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except FileNotFoundError:
            print("Data file not found.")
            return None

    def plot_issues_over_time(self, df):
        """Line chart of issues count over time."""
        if 'timestamp' not in df.columns: return None
        
        df_daily = df.groupby(df['timestamp'].dt.date).size().reset_index(name='count')
        fig = px.line(df_daily, x='timestamp', y='count', title='Number of Issues Over Time')
        return fig

    def plot_top_organizations(self, df, top_n=10):
        """Bar chart of top organizations."""
        top_orgs = df['organization'].value_counts().nlargest(top_n).reset_index()
        top_orgs.columns = ['organization', 'count']
        fig = px.bar(top_orgs, x='organization', y='count', title=f'Top {top_n} Organizations')
        return fig

    def plot_resolution_time_distribution(self, df):
        """Histogram of resolution time."""
        if 'resolution_time' not in df.columns and 'resolution_time_hrs' not in df.columns:
             # Calculate if missing (re-doing logic from modeler slightly if needed for viz)
             if 'last_activity' in df.columns and 'timestamp' in df.columns:
                 df['last_activity'] = pd.to_datetime(df['last_activity'])
                 df['resolution_time_hrs'] = (df['last_activity'] - df['timestamp']).dt.total_seconds() / 3600
        
        target_col = 'resolution_time_hrs' if 'resolution_time_hrs' in df.columns else 'resolution_time'
        
        if target_col in df.columns:
            # Filter outliers for visualization
            q_low = df[target_col].quantile(0.01)
            q_high = df[target_col].quantile(0.99)
            df_filtered = df[(df[target_col] > q_low) & (df[target_col] < q_high)]
            
            fig = px.histogram(df_filtered, x=target_col, nbins=50, title='Resolution Time Distribution (Hours)')
            return fig
        return None

    def generate_report(self):
        df = self.load_data()
        if df is not None:
            print("Generating visualizations...")
            fig1 = self.plot_issues_over_time(df)
            if fig1: fig1.show()
            
            fig2 = self.plot_top_organizations(df)
            if fig2: fig2.show()
            
            fig3 = self.plot_resolution_time_distribution(df)
            if fig3: fig3.show()

if __name__ == "__main__":
    viz = Visualizer()
    viz.generate_report()
