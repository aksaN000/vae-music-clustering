"""
Model Comparison & Analysis System
Q1 Standard: Statistical analysis and publication-ready tables
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ModelComparator:
    """Statistical comparison and analysis of models"""
    
    def __init__(self, results_dir="results/evaluations"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path("results/comparisons")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Comparator initialized. Output: {self.output_dir}")
    
    def load_results(self):
        """Load evaluation results including baselines"""
        
        # Load VAE results
        csv_path = self.results_dir / "clustering_metrics.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"VAE results not found: {csv_path}")
        
        df_vae = pd.read_csv(csv_path)
        print(f"Loaded {len(df_vae)} VAE evaluation results")
        
        # Load baseline results
        baseline_path = self.results_dir / "baseline_clustering_metrics.csv"
        if baseline_path.exists():
            df_baseline = pd.read_csv(baseline_path)
            print(f"Loaded {len(df_baseline)} baseline evaluation results")
            
            # Merge
            df = pd.concat([df_vae, df_baseline], ignore_index=True)
            print(f"Total: {len(df)} results (VAE + Baselines)")
        else:
            print("Warning: Baseline results not found, continuing with VAEs only")
            df = df_vae
        
        return df
    
    def create_summary_table(self, df):
        """Create comprehensive summary table"""
        
        # Aggregate by model and task
        summary = df.groupby(['model', 'task']).agg({
            'silhouette': ['mean', 'std'],
            'ari': ['mean', 'std'],
            'nmi': ['mean', 'std'],
            'v_measure': ['mean', 'std'],
            'calinski_harabasz': ['mean', 'std'],
            'davies_bouldin': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        
        # Save
        output_path = self.output_dir / "summary_table.csv"
        summary.to_csv(output_path, index=False)
        print(f"  * Summary table saved: {output_path.name}")
        
        return summary
    
    def pairwise_comparison(self, df):
        """Statistical pairwise comparisons between models"""
        
        models = df['model'].unique()
        metrics = ['silhouette', 'ari', 'nmi', 'v_measure']
        
        comparisons = []
        
        for metric in metrics:
            for i, model1 in enumerate(models):
                for model2 in models[i+1:]:
                    
                    scores1 = df[df['model'] == model1][metric].values
                    scores2 = df[df['model'] == model2][metric].values
                    
                    # T-test
                    t_stat, p_value = stats.ttest_ind(scores1, scores2)
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt((scores1.std()**2 + scores2.std()**2) / 2)
                    cohens_d = (scores1.mean() - scores2.mean()) / pooled_std if pooled_std > 0 else 0
                    
                    comparisons.append({
                        'metric': metric,
                        'model_1': model1,
                        'model_2': model2,
                        'mean_diff': scores1.mean() - scores2.mean(),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'cohens_d': cohens_d
                    })
        
        comp_df = pd.DataFrame(comparisons)
        
        # Save
        output_path = self.output_dir / "pairwise_comparisons.csv"
        comp_df.to_csv(output_path, index=False)
        print(f"  * Pairwise comparisons saved: {output_path.name}")
        
        return comp_df
    
    def task_analysis(self, df):
        """Analyze performance by task"""
        
        task_summary = []
        
        for task in df['task'].unique():
            task_df = df[df['task'] == task]
            
            summary = {
                'task': task,
                'n_samples': len(task_df),
                'best_model_ari': task_df.loc[task_df['ari'].idxmax(), 'model'],
                'best_ari': task_df['ari'].max(),
                'best_model_nmi': task_df.loc[task_df['nmi'].idxmax(), 'model'],
                'best_nmi': task_df['nmi'].max(),
                'best_model_silhouette': task_df.loc[task_df['silhouette'].idxmax(), 'model'],
                'best_silhouette': task_df['silhouette'].max(),
                'avg_ari': task_df['ari'].mean(),
                'avg_nmi': task_df['nmi'].mean(),
                'avg_silhouette': task_df['silhouette'].mean(),
            }
            
            task_summary.append(summary)
        
        task_df_summary = pd.DataFrame(task_summary)
        
        # Save
        output_path = self.output_dir / "task_analysis.csv"
        task_df_summary.to_csv(output_path, index=False)
        print(f"  * Task analysis saved: {output_path.name}")
        
        return task_df_summary
    
    def method_analysis(self, df):
        """Analyze clustering methods"""
        
        method_summary = df.groupby('method').agg({
            'silhouette': ['mean', 'std', 'min', 'max'],
            'ari': ['mean', 'std', 'min', 'max'],
            'nmi': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        method_summary.columns = ['_'.join(col).strip() for col in method_summary.columns.values]
        method_summary = method_summary.reset_index()
        
        # Save
        output_path = self.output_dir / "method_analysis.csv"
        method_summary.to_csv(output_path, index=False)
        print(f"  * Method analysis saved: {output_path.name}")
        
        return method_summary
    
    def create_latex_table(self, df):
        """Generate LaTeX table for publication"""
        
        # Create pivot table for main results
        pivot = df.pivot_table(
            values=['silhouette', 'ari', 'nmi', 'v_measure'],
            index='model',
            aggfunc='mean'
        ).round(4)
        
        # Generate LaTeX
        latex_str = pivot.to_latex(
            column_format='l' + 'c' * len(pivot.columns),
            bold_rows=True,
            caption='Model Performance Comparison',
            label='tab:model_comparison'
        )
        
        # Save
        output_path = self.output_dir / "table_latex.tex"
        with open(output_path, 'w') as f:
            f.write(latex_str)
        
        print(f"  * LaTeX table saved: {output_path.name}")
    
    def create_markdown_report(self, df, summary_df, task_df, method_df):
        """Create comprehensive markdown report"""
        
        output_path = self.output_dir / "comparison_report.md"
        
        with open(output_path, 'w') as f:
            f.write("# Model Comparison Report\n\n")
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Models**: {df['model'].nunique()}\n")
            f.write(f"- **Total Experiments**: {len(df)}\n")
            f.write(f"- **Tasks**: {', '.join(df['task'].unique())}\n")
            f.write(f"- **Clustering Methods**: {', '.join(df['method'].unique())}\n\n")
            
            f.write("## Overall Performance Rankings\n\n")
            
            # Calculate overall scores
            model_scores = df.groupby('model').agg({
                'silhouette': 'mean',
                'ari': 'mean',
                'nmi': 'mean',
                'v_measure': 'mean'
            })
            model_scores['overall'] = model_scores.mean(axis=1)
            model_scores = model_scores.sort_values('overall', ascending=False)
            
            f.write("| Rank | Model | Overall Score | Silhouette | ARI | NMI | V-Measure |\n")
            f.write("|------|-------|---------------|------------|-----|-----|------------|\n")
            
            for rank, (model, row) in enumerate(model_scores.iterrows(), 1):
                f.write(f"| {rank} | {model} | {row['overall']:.4f} | "
                       f"{row['silhouette']:.4f} | {row['ari']:.4f} | "
                       f"{row['nmi']:.4f} | {row['v_measure']:.4f} |\n")
            
            f.write("\n## Task-Specific Performance\n\n")
            
            for task in task_df['task'].unique():
                task_row = task_df[task_df['task'] == task].iloc[0]
                f.write(f"### {task.capitalize()} Clustering\n\n")
                f.write(f"- **Best ARI**: {task_row['best_ari']:.4f} ({task_row['best_model_ari']})\n")
                f.write(f"- **Best NMI**: {task_row['best_nmi']:.4f} ({task_row['best_model_nmi']})\n")
                f.write(f"- **Best Silhouette**: {task_row['best_silhouette']:.4f} ({task_row['best_model_silhouette']})\n\n")
            
            f.write("## Clustering Method Comparison\n\n")
            f.write("| Method | Avg Silhouette | Avg ARI | Avg NMI |\n")
            f.write("|--------|----------------|---------|----------|\n")
            
            for _, row in method_df.iterrows():
                f.write(f"| {row['method']} | {row['silhouette_mean']:.4f} | "
                       f"{row['ari_mean']:.4f} | {row['nmi_mean']:.4f} |\n")
            
            f.write("\n## Key Findings\n\n")
            
            # Best overall model
            best_model = model_scores.index[0]
            f.write(f"1. **Best Overall Model**: {best_model} "
                   f"(Overall Score: {model_scores.iloc[0]['overall']:.4f})\n")
            
            # Best for each task
            for task in df['task'].unique():
                task_df_filtered = df[df['task'] == task]
                best_model_task = task_df_filtered.groupby('model')['ari'].mean().idxmax()
                best_ari = task_df_filtered.groupby('model')['ari'].mean().max()
                f.write(f"2. **Best for {task.capitalize()}**: {best_model_task} "
                       f"(ARI: {best_ari:.4f})\n")
            
            # Best clustering method
            best_method = method_df.loc[method_df['ari_mean'].idxmax(), 'method']
            f.write(f"3. **Best Clustering Method**: {best_method}\n")
        
        print(f"  * Markdown report saved: {output_path.name}")


def main():
    print("\n" + "="*80)
    print("MODEL COMPARISON & ANALYSIS")
    print("="*80)
    
    comparator = ModelComparator()
    
    # Load results
    print(f"\n{'='*60}")
    print("Loading Results")
    print('='*60)
    df = comparator.load_results()
    
    # Generate analyses
    print(f"\n{'='*60}")
    print("Generating Comparisons")
    print('='*60)
    
    summary_df = comparator.create_summary_table(df)
    comp_df = comparator.pairwise_comparison(df)
    task_df = comparator.task_analysis(df)
    method_df = comparator.method_analysis(df)
    comparator.create_latex_table(df)
    comparator.create_markdown_report(df, summary_df, task_df, method_df)
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"All analyses saved in: {comparator.output_dir}")
    print('='*80)


if __name__ == "__main__":
    main()
