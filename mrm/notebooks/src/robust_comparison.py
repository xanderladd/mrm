"""Compare denoising results across all models"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple

def load_all_results(cache_dir='cache') -> Dict:
    """Load all denoising CV results from cache"""
    results = {}
    
    # Pattern: cache/{model}_cv_denoising_{corruption}_{level}
    for path in Path(cache_dir).glob('*_cv_denoising_*/cv_results.json'):
        with open(path) as f:
            data = json.load(f)
            
        model = data['model_type']
        corruption = data['corruption_type']
        level = data['corruption_level']
        
        if model not in results:
            results[model] = {}
        if corruption not in results[model]:
            results[model][corruption] = {}
        
        results[model][corruption][level] = data
    
    return results

def create_comparison_plots(results: Dict) -> plt.Figure:
    """Create comprehensive comparison plots"""
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    models = sorted(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(models)))
    
    # 1. Clean baseline performance
    ax1 = fig.add_subplot(gs[0, 0])
    baseline_data = []
    
    for model in models:
        if 'none' in results[model] and 0.0 in results[model]['none']:
            r2 = results[model]['none'][0.0]['mean_test_r2']
            std = results[model]['none'][0.0]['std_test_r2']
            baseline_data.append((model.upper(), r2, std))
    
    if baseline_data:
        names, r2s, stds = zip(*baseline_data)
        x_pos = np.arange(len(names))
        bars = ax1.bar(x_pos, r2s, yerr=stds, capsize=5, alpha=0.7)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.set_ylabel('R²')
        ax1.set_title('Baseline Performance (Clean Data)')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, 1])
        
        # Color best performer
        best_idx = np.argmax(r2s)
        bars[best_idx].set_color('green')
        bars[best_idx].set_alpha(1.0)
    
    # 2. Gaussian noise robustness (R²)
    ax2 = fig.add_subplot(gs[0, 1])
    for i, model in enumerate(models):
        if 'noise' in results[model]:
            noise_data = []
            
            # Add baseline point
            if 'none' in results[model] and 0.0 in results[model]['none']:
                noise_data.append((0.0, 
                                 results[model]['none'][0.0]['mean_test_r2'],
                                 results[model]['none'][0.0]['std_test_r2']))
            
            # Add noise points
            for level in sorted(results[model]['noise'].keys()):
                noise_data.append((level,
                                 results[model]['noise'][level]['mean_test_r2'],
                                 results[model]['noise'][level]['std_test_r2']))
            
            if noise_data:
                levels, means, stds = zip(*noise_data)
                ax2.errorbar(levels, means, yerr=stds, marker='o', 
                           label=model.upper(), color=colors[i], 
                           capsize=3, alpha=0.8, linewidth=2)
    
    ax2.set_xlabel('Noise Level (σ)')
    ax2.set_ylabel('R²')
    ax2.set_title('Gaussian Noise Robustness')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-0.02, max(0.3, ax2.get_xlim()[1])])
    ax2.set_ylim([0, 1])
    
    # 3. Noise degradation rate
    ax3 = fig.add_subplot(gs[0, 2])
    degradation_data = []
    
    for model in models:
        if 'none' in results[model] and 'noise' in results[model]:
            baseline_r2 = results[model]['none'][0.0]['mean_test_r2']
            if 0.2 in results[model]['noise']:
                noisy_r2 = results[model]['noise'][0.2]['mean_test_r2']
                degradation = (baseline_r2 - noisy_r2) / baseline_r2 * 100
                degradation_data.append((model.upper(), degradation))
    
    if degradation_data:
        names, degradations = zip(*degradation_data)
        x_pos = np.arange(len(names))
        bars = ax3.bar(x_pos, degradations, alpha=0.7)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(names, rotation=45, ha='right')
        ax3.set_ylabel('% Performance Drop')
        ax3.set_title('Degradation under Noise (σ=0.2)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Color best (least degradation)
        best_idx = np.argmin(degradations)
        bars[best_idx].set_color('green')
        bars[best_idx].set_alpha(1.0)
    
    # 4. Masking - Overall performance
    ax4 = fig.add_subplot(gs[1, 0])
    for i, model in enumerate(models):
        if 'mask' in results[model]:
            mask_data = []
            
            # Add baseline
            if 'none' in results[model] and 0.0 in results[model]['none']:
                mask_data.append((0.0,
                                results[model]['none'][0.0]['mean_test_r2'],
                                results[model]['none'][0.0]['std_test_r2']))
            
            # Add mask points
            for ratio in sorted(results[model]['mask'].keys()):
                mask_data.append((ratio,
                                results[model]['mask'][ratio]['mean_test_r2'],
                                results[model]['mask'][ratio]['std_test_r2']))
            
            if mask_data:
                ratios, means, stds = zip(*mask_data)
                ax4.errorbar(ratios, means, yerr=stds, marker='s',
                           label=model.upper(), color=colors[i],
                           capsize=3, alpha=0.8, linewidth=2)
    
    ax4.set_xlabel('Mask Ratio')
    ax4.set_ylabel('R² (Overall)')
    ax4.set_title('Masking: Overall Performance')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([-0.02, max(0.3, ax4.get_xlim()[1])])
    ax4.set_ylim([0, 1])
    
    # 5. Masking - Masked neurons only (KEY METRIC)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor('#fff8dc')  # Highlight this plot
    
    for i, model in enumerate(models):
        if 'mask' in results[model]:
            masked_data = []
            
            for ratio in sorted(results[model]['mask'].keys()):
                if ratio > 0 and 'mean_masked_r2' in results[model]['mask'][ratio]:
                    masked_data.append((ratio,
                                      results[model]['mask'][ratio]['mean_masked_r2'],
                                      results[model]['mask'][ratio].get('std_masked_r2', 0)))
            
            if masked_data:
                ratios, means, stds = zip(*masked_data)
                ax5.errorbar(ratios, means, yerr=stds, marker='^',
                           label=model.upper(), color=colors[i],
                           capsize=3, linewidth=2.5, markersize=8)
    
    ax5.set_xlabel('Mask Ratio')
    ax5.set_ylabel('R² (Masked Neurons Only)')
    ax5.set_title('🔑 KEY METRIC: Masked Neuron Reconstruction', fontweight='bold', fontsize=11)
    ax5.legend(loc='best', fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, max(0.3, ax5.get_xlim()[1])])
    
    # Add interpretation line
    ax5.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Good reconstruction threshold')
    ax5.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Moderate reconstruction')
    
    # 6. Model ranking
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Calculate average rank across all metrics
    rankings = calculate_model_rankings(results)
    
    if rankings:
        models_sorted = sorted(rankings.items(), key=lambda x: x[1])
        names, ranks = zip(*models_sorted)
        
        x_pos = np.arange(len(names))
        bars = ax6.barh(x_pos, ranks, alpha=0.7)
        ax6.set_yticks(x_pos)
        ax6.set_yticklabels([n.upper() for n in names])
        ax6.set_xlabel('Average Rank (lower is better)')
        ax6.set_title('Overall Model Ranking')
        ax6.grid(True, alpha=0.3, axis='x')
        
        # Color code
        bars[0].set_color('gold')
        if len(bars) > 1:
            bars[1].set_color('silver')
        if len(bars) > 2:
            bars[2].set_color('#CD7F32')  # Bronze
    
    # 7. Summary table
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('tight')
    ax7.axis('off')
    
    # Create comprehensive summary
    summary_data = []
    for model in models:
        row = [model.upper()]
        
        # Clean R²
        if 'none' in results[model] and 0.0 in results[model]['none']:
            val = results[model]['none'][0.0]['mean_test_r2']
            std = results[model]['none'][0.0]['std_test_r2']
            row.append(f"{val:.3f}±{std:.3f}")
        else:
            row.append("N/A")
        
        # Noise σ=0.2
        if 'noise' in results[model] and 0.2 in results[model]['noise']:
            val = results[model]['noise'][0.2]['mean_test_r2']
            std = results[model]['noise'][0.2]['std_test_r2']
            row.append(f"{val:.3f}±{std:.3f}")
        else:
            row.append("N/A")
        
        # Mask 20% - Overall
        if 'mask' in results[model] and 0.2 in results[model]['mask']:
            val = results[model]['mask'][0.2]['mean_test_r2']
            std = results[model]['mask'][0.2]['std_test_r2']
            row.append(f"{val:.3f}±{std:.3f}")
        else:
            row.append("N/A")
        
        # Mask 20% - Masked neurons only
        if 'mask' in results[model] and 0.2 in results[model]['mask']:
            if 'mean_masked_r2' in results[model]['mask'][0.2]:
                val = results[model]['mask'][0.2]['mean_masked_r2']
                std = results[model]['mask'][0.2].get('std_masked_r2', 0)
                row.append(f"{val:.3f}±{std:.3f}")
            else:
                row.append("N/A")
        else:
            row.append("N/A")
        
        summary_data.append(row)
    
    columns = ['Model', 'Clean R²', 'Noise σ=0.2', 'Mask 20%\n(Overall)', 'Mask 20%\n(Masked Only)']
    table = ax7.table(cellText=summary_data, colLabels=columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    # Highlight best in each column
    for col in range(1, 5):
        values = []
        for row in range(len(summary_data)):
            try:
                # Extract mean value from "mean±std" format
                val_str = summary_data[row][col].split('±')[0]
                values.append(float(val_str))
            except:
                values.append(-np.inf)
        
        if values and max(values) > -np.inf:
            best_idx = np.argmax(values)
            table[(best_idx + 1, col)].set_facecolor('#90EE90')
            table[(best_idx + 1, col)].set_text_props(weight='bold')
    
    # Style header
    for col in range(len(columns)):
        table[(0, col)].set_facecolor('#D3D3D3')
        table[(0, col)].set_text_props(weight='bold')
    
    ax7.set_title('Summary Table (Mean ± Std)', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('Denoising Cross-Validation Results', fontsize=16, fontweight='bold', y=0.98)
    
    return fig

def calculate_model_rankings(results: Dict) -> Dict[str, float]:
    """Calculate average ranking across all metrics"""
    rankings = {}
    metrics_count = 0
    
    # Rank models for each metric
    for metric_configs in [
        ('none', 0.0, 'mean_test_r2'),
        ('noise', 0.2, 'mean_test_r2'),
        ('mask', 0.2, 'mean_test_r2'),
        ('mask', 0.2, 'mean_masked_r2')
    ]:
        corruption, level, metric_key = metric_configs
        scores = {}
        
        for model in results:
            if corruption in results[model] and level in results[model][corruption]:
                if metric_key in results[model][corruption][level]:
                    scores[model] = results[model][corruption][level][metric_key]
        
        if scores:
            metrics_count += 1
            sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (model, _) in enumerate(sorted_models, 1):
                if model not in rankings:
                    rankings[model] = 0
                rankings[model] += rank
    
    # Average ranks
    for model in rankings:
        rankings[model] /= metrics_count
    
    return rankings

def print_detailed_summary(results: Dict):
    """Print detailed summary with insights"""
    
    print("\n" + "="*80)
    print("DENOISING CROSS-VALIDATION DETAILED SUMMARY")
    print("="*80)
    
    # 1. Clean baseline
    print("\n1. BASELINE PERFORMANCE (Clean Data):")
    print("-"*40)
    clean_scores = {}
    for model in results:
        if 'none' in results[model] and 0.0 in results[model]['none']:
            clean_scores[model] = (results[model]['none'][0.0]['mean_test_r2'],
                                  results[model]['none'][0.0]['std_test_r2'])
    
    if clean_scores:
        sorted_models = sorted(clean_scores.items(), key=lambda x: x[1][0], reverse=True)
        for model, (mean, std) in sorted_models:
            print(f"  {model.upper():12s}: R² = {mean:.3f} ± {std:.3f}")
    
    # 2. Gaussian noise denoising
    print("\n2. GAUSSIAN NOISE DENOISING (σ=0.2):")
    print("-"*40)
    noise_scores = {}
    for model in results:
        if 'noise' in results[model] and 0.2 in results[model]['noise']:
            noise_scores[model] = (results[model]['noise'][0.2]['mean_test_r2'],
                                  results[model]['noise'][0.2]['std_test_r2'])
    
    if noise_scores:
        sorted_models = sorted(noise_scores.items(), key=lambda x: x[1][0], reverse=True)
        for model, (mean, std) in sorted_models:
            print(f"  {model.upper():12s}: R² = {mean:.3f} ± {std:.3f}")
            
            # Calculate degradation
            if model in clean_scores:
                degradation = (clean_scores[model][0] - mean) / clean_scores[model][0] * 100
                print(f"                 (↓ {degradation:.1f}% from baseline)")
    
    # 3. Masked neuron reconstruction (KEY METRIC)
    print("\n3. MASKED NEURON RECONSTRUCTION (20% masked) - 🔑 KEY METRIC:")
    print("-"*40)
    masked_scores = {}
    for model in results:
        if 'mask' in results[model] and 0.2 in results[model]['mask']:
            if 'mean_masked_r2' in results[model]['mask'][0.2]:
                masked_scores[model] = (results[model]['mask'][0.2]['mean_masked_r2'],
                                       results[model]['mask'][0.2].get('std_masked_r2', 0))
    
    if masked_scores:
        sorted_models = sorted(masked_scores.items(), key=lambda x: x[1][0], reverse=True)
        for model, (mean, std) in sorted_models:
            print(f"  {model.upper():12s}: R² = {mean:.3f} ± {std:.3f}")
        
        print("\n  INTERPRETATION:")
        print("  " + "-"*35)
        best_model, (best_score, _) = sorted_models[0]
        
        if best_score > 0.6:
            print(f"  ✅ EXCELLENT: {best_model.upper()} achieves R²={best_score:.3f}")
            print("     → Strong evidence of learning neural correlations")
            print("     → Model captures underlying neural dynamics")
        elif best_score > 0.4:
            print(f"  ✓ GOOD: {best_model.upper()} achieves R²={best_score:.3f}")
            print("     → Moderate evidence of learning beyond memorization")
            print("     → Some neural structure captured")
        elif best_score > 0.2:
            print(f"  ⚠️  WEAK: {best_model.upper()} only achieves R²={best_score:.3f}")
            print("     → Limited reconstruction of masked neurons")
            print("     → May rely heavily on identity mapping")
        else:
            print(f"  ❌ POOR: Best model only achieves R²={best_score:.3f}")
            print("     → Models fail to reconstruct masked neurons")
            print("     → Strong indication of memorization/identity mapping")
    
    # 4. Model rankings
    print("\n4. OVERALL MODEL RANKINGS:")
    print("-"*40)
    rankings = calculate_model_rankings(results)
    if rankings:
        sorted_rankings = sorted(rankings.items(), key=lambda x: x[1])
        medals = ['🥇', '🥈', '🥉']
        for i, (model, avg_rank) in enumerate(sorted_rankings):
            medal = medals[i] if i < len(medals) else '  '
            print(f"  {medal} {i+1}. {model.upper():12s} (avg rank: {avg_rank:.2f})")
    
    # 5. Key findings
    print("\n5. KEY FINDINGS:")
    print("-"*40)
    
    # Check if Kalman has unfair advantage
    if 'kalman' in results and 'noise' in results.get('kalman', {}):
        kalman_noise_perf = results['kalman']['noise'].get(0.2, {}).get('mean_test_r2', 0)
        other_noise_perfs = []
        for model in results:
            if model != 'kalman' and 'noise' in results[model]:
                if 0.2 in results[model]['noise']:
                    other_noise_perfs.append(results[model]['noise'][0.2]['mean_test_r2'])
        
        if other_noise_perfs and kalman_noise_perf > max(other_noise_perfs) * 1.2:
            print("  ⚠️  Kalman filter shows suspiciously good noise performance")
            print("     → May have unfair advantage due to Gaussian noise assumption")
            print("     → Consider results with caution")
    
    # Identify models that can truly denoise
    print("\n  Models that successfully denoise/interpolate:")
    for model in results:
        if 'mask' in results[model] and 0.2 in results[model]['mask']:
            if 'mean_masked_r2' in results[model]['mask'][0.2]:
                r2 = results[model]['mask'][0.2]['mean_masked_r2']
                if r2 > 0.3:
                    print(f"    ✓ {model.upper()}: Can reconstruct unseen neurons (R²={r2:.3f})")
    
    print("\n" + "="*80)

def save_results_to_csv(results: Dict, output_path: str = "denoising_results.csv"):
    """Save results to CSV for further analysis"""
    
    rows = []
    for model in results:
        for corruption in results[model]:
            for level in results[model][corruption]:
                data = results[model][corruption][level]
                row = {
                    'model': model,
                    'corruption': corruption,
                    'level': level,
                    'mean_r2': data['mean_test_r2'],
                    'std_r2': data['std_test_r2'],
                    'mean_mse': data['mean_test_mse'],
                    'std_mse': data['std_test_mse']
                }
                
                # Add masked metrics if present
                if 'mean_masked_r2' in data:
                    row['mean_masked_r2'] = data['mean_masked_r2']
                    row['std_masked_r2'] = data.get('std_masked_r2', 0)
                    row['mean_masked_mse'] = data.get('mean_masked_mse', 0)
                    row['std_masked_mse'] = data.get('std_masked_mse', 0)
                
                rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    return df

def main():
    """Main analysis function"""
    
    # Load results
    print("Loading denoising results...")
    results = load_all_results()
    
    if not results:
        print("❌ No results found! Run denoising experiments first:")
        print("   python cross_validate_denoising.py configs/model.json")
        return
    
    print(f"Found results for {len(results)} models")
    
    # Print summary
    print_detailed_summary(results)
    
    # Create plots
    print("\nGenerating comparison plots...")
    fig = create_comparison_plots(results)
    
    # Save plots
    output_path = "denoising_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plots saved to {output_path}")
    
    # Save CSV
    df = save_results_to_csv(results)
    
    # Show plots
    plt.show()

if __name__ == '__main__':
    main()