#!/usr/bin/env python3
"""
Alpha ICU GUI - Display alphas with low production and self correlation that pass tests
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime
import threading

from alpha_fetcher import AlphaFetcher
from alpha_analyzer import AlphaAnalyzer, AlphaMetrics
from correlation_checker import CorrelationChecker

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlphaICU_GUI:
    """GUI for Alpha ICU system"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Alpha ICU - Low Correlation Alpha Finder")
        self.root.geometry("1400x900")
        
        # Initialize components
        self.fetcher = None
        self.analyzer = None
        self.correlation_checker = None
        self.successful_alphas = []
        self.correlation_data = {}
        
        self.setup_ui()
        self.initialize_components()
    
    def setup_ui(self):
        """Set up the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Alpha ICU - Low Correlation Alpha Finder", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Analysis Controls", padding="10")
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.columnconfigure(1, weight=1)
        
        # Max alphas input
        ttk.Label(control_frame, text="Max Alphas:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.max_alphas_var = tk.StringVar(value="50")
        max_alphas_entry = ttk.Entry(control_frame, textvariable=self.max_alphas_var, width=10)
        max_alphas_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        # Days back input
        ttk.Label(control_frame, text="Days Back:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.days_back_var = tk.StringVar(value="7")
        days_back_entry = ttk.Entry(control_frame, textvariable=self.days_back_var, width=10)
        days_back_entry.grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        
        # Max correlation threshold
        ttk.Label(control_frame, text="Max Prod Correlation:").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        self.max_corr_var = tk.StringVar(value="0.3")
        max_corr_entry = ttk.Entry(control_frame, textvariable=self.max_corr_var, width=10)
        max_corr_entry.grid(row=0, column=5, sticky=tk.W, padx=(0, 20))
        
        # Run analysis button
        self.run_button = ttk.Button(control_frame, text="Run Analysis", command=self.run_analysis)
        self.run_button.grid(row=0, column=6, padx=(10, 0))
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready")
        progress_label = ttk.Label(control_frame, textvariable=self.progress_var)
        progress_label.grid(row=1, column=0, columnspan=7, pady=(10, 0))
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Summary tab
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="Summary")
        
        # Alphas tab
        self.alphas_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.alphas_frame, text="Low Correlation Alphas")
        
        # Setup summary tab
        self.setup_summary_tab()
        
        # Setup alphas tab
        self.setup_alphas_tab()
    
    def setup_summary_tab(self):
        """Set up the summary tab"""
        # Summary text widget
        self.summary_text = scrolledtext.ScrolledText(self.summary_frame, height=20, width=100)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Export button
        export_button = ttk.Button(self.summary_frame, text="Export Results", command=self.export_results)
        export_button.pack(pady=10)
    
    def setup_alphas_tab(self):
        """Set up the alphas tab with treeview"""
        # Create treeview for alphas
        columns = ("Alpha ID", "Sharpe", "Margin", "Prod Corr", "Self Corr", "Region", "Universe", "Status")
        self.alphas_tree = ttk.Treeview(self.alphas_frame, columns=columns, show="headings", height=20)
        
        # Configure columns
        for col in columns:
            self.alphas_tree.heading(col, text=col)
            self.alphas_tree.column(col, width=100, anchor=tk.CENTER)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(self.alphas_frame, orient=tk.VERTICAL, command=self.alphas_tree.yview)
        h_scrollbar = ttk.Scrollbar(self.alphas_frame, orient=tk.HORIZONTAL, command=self.alphas_tree.xview)
        self.alphas_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack widgets
        self.alphas_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X, padx=(10, 0))
        
        # Bind double-click event
        self.alphas_tree.bind("<Double-1>", self.on_alpha_double_click)
    
    def initialize_components(self):
        """Initialize the Alpha ICU components"""
        try:
            self.progress_var.set("Initializing components...")
            self.root.update()
            
            # Initialize fetcher
            self.fetcher = AlphaFetcher("credential.txt")
            
            # Initialize analyzer with new criteria
            self.analyzer = AlphaAnalyzer(
                min_sharpe=1.2,  # Sharpe > 1.2
                min_margin=0.0008,  # Margin > 8 bps
                max_prod_correlation=0.7  # Hard constraint
            )
            
            # Initialize correlation checker
            self.correlation_checker = CorrelationChecker(
                high_correlation_threshold=0.5,
                medium_correlation_threshold=0.2,
                max_high_correlations=1000
            )
            
            self.progress_var.set("Components initialized successfully")
            logger.info("GUI components initialized successfully")
            
        except Exception as e:
            self.progress_var.set(f"Initialization failed: {str(e)}")
            messagebox.showerror("Initialization Error", f"Failed to initialize components: {str(e)}")
            logger.error(f"Failed to initialize components: {e}")
    
    def run_analysis(self):
        """Run the alpha analysis in a separate thread"""
        if not self.fetcher or not self.analyzer or not self.correlation_checker:
            messagebox.showerror("Error", "Components not initialized")
            return
        
        # Disable button during analysis
        self.run_button.config(state="disabled")
        self.progress_var.set("Starting analysis...")
        
        # Run in separate thread
        thread = threading.Thread(target=self._run_analysis_thread)
        thread.daemon = True
        thread.start()
    
    def _run_analysis_thread(self):
        """Run analysis in background thread"""
        try:
            # Get parameters
            max_alphas = int(self.max_alphas_var.get())
            days_back = int(self.days_back_var.get())
            max_corr_threshold = float(self.max_corr_var.get())
            
            self.progress_var.set("Fetching alphas...")
            self.root.update()
            
            # Fetch alphas with performance filters
            alphas = self.fetcher.fetch_all_alphas(
                max_alphas=max_alphas,
                status="UNSUBMITTED,IS_FAIL",
                min_sharpe=1.2,      # Filter for Sharpe > 1.2
                min_fitness=1.0,     # Filter for fitness > 1
                min_margin=0.0005    # Filter for margin > 0.0005 (5 bps)
            )
            
            if not alphas:
                self.progress_var.set("No alphas found")
                self.root.after(0, lambda: messagebox.showinfo("No Results", "No alphas found with the specified criteria"))
                return
            
            self.progress_var.set("Pre-filtering alphas for basic criteria...")
            self.root.update()
            
            # Pre-filter alphas for basic criteria (avoid unnecessary correlation checks)
            candidate_alphas = []
            for alpha_data in alphas:
                try:
                    metrics = self.analyzer.extract_alpha_metrics(alpha_data)
                    # Check basic criteria without correlation
                    is_successful, reasons = self.analyzer.is_successful_alpha(metrics, None)
                    if is_successful:
                        candidate_alphas.append((alpha_data, metrics))
                except Exception as e:
                    logger.error(f"Error processing alpha {alpha_data.get('id', 'unknown')}: {e}")
                    continue
            
            self.progress_var.set(f"Found {len(candidate_alphas)} candidate alphas out of {len(alphas)} total, checking correlations...")
            self.root.update()
            
            # Check correlations only for candidate alphas with rate limiting
            correlation_analyses = []
            for i, (alpha_data, metrics) in enumerate(candidate_alphas):
                alpha_id = alpha_data.get('id', '')
                self.progress_var.set(f"Checking correlations {i+1}/{len(candidate_alphas)}: {alpha_id}")
                self.root.update()
                
                correlation_data = self.fetcher.get_correlation_data(alpha_id)
                analysis = self.correlation_checker.analyze_correlation_data(alpha_id, correlation_data)
                correlation_analyses.append(analysis)
                
                # Store correlation data
                self.correlation_data[alpha_id] = analysis
                
                # Add delay between requests to avoid rate limiting (except for last request)
                if i < len(candidate_alphas) - 1:
                    time.sleep(2.0)  # 2 second delay between requests to respect 60/min limit
            
            self.progress_var.set("Final filtering with correlation constraints...")
            self.root.update()
            
            # Final filtering with correlation constraints
            successful_alphas = []
            for (alpha_data, metrics), correlation_analysis in zip(candidate_alphas, correlation_analyses):
                try:
                    max_correlation = correlation_analysis.max_correlation
                    
                    # Check if alpha passes criteria with correlation
                    is_successful, reasons = self.analyzer.is_successful_alpha(metrics, max_correlation)
                    
                    # Additional filter: only include alphas with low production correlation
                    if is_successful and max_correlation <= max_corr_threshold:
                        successful_alphas.append({
                            'metrics': metrics,
                            'correlation_analysis': correlation_analysis,
                            'raw_data': alpha_data
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing alpha {alpha_data.get('id', 'unknown')}: {e}")
                    continue
            
            self.successful_alphas = successful_alphas
            
            # Update UI in main thread
            self.root.after(0, self._update_results_ui)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.progress_var.set(error_msg)
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", error_msg))
            logger.error(f"Analysis failed: {e}")
        finally:
            self.root.after(0, lambda: self.run_button.config(state="normal"))
    
    def _update_results_ui(self):
        """Update the results UI with analysis results"""
        try:
            # Update summary
            self._update_summary()
            
            # Update alphas table
            self._update_alphas_table()
            
            self.progress_var.set(f"Analysis complete: {len(self.successful_alphas)} low correlation alphas found")
            
        except Exception as e:
            logger.error(f"Error updating UI: {e}")
            messagebox.showerror("UI Update Error", f"Failed to update results: {str(e)}")
    
    def _update_summary(self):
        """Update the summary tab"""
        self.summary_text.delete(1.0, tk.END)
        
        if not self.successful_alphas:
            self.summary_text.insert(tk.END, "No low correlation alphas found that pass the criteria.\n\n")
            self.summary_text.insert(tk.END, "Criteria used:\n")
            self.summary_text.insert(tk.END, "- API Response: FAIL=reject, WARNING=accept\n")
            self.summary_text.insert(tk.END, "- Sharpe ratio > 1.2\n")
            self.summary_text.insert(tk.END, "- Margin > 8 bps (0.0008)\n")
            self.summary_text.insert(tk.END, "- Production correlation <= 0.7 (hard constraint)\n")
            self.summary_text.insert(tk.END, f"- Max production correlation <= {self.max_corr_var.get()}\n")
            return
        
        # Generate summary
        summary = f"Alpha ICU Analysis Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += "=" * 60 + "\n\n"
        
        summary += f"Total low correlation alphas found: {len(self.successful_alphas)}\n\n"
        
        # Criteria used
        summary += "Criteria used:\n"
        summary += "- API Response: FAIL=reject, WARNING=accept\n"
        summary += "- Sharpe ratio > 1.2\n"
        summary += "- Margin > 8 bps (0.0008)\n"
        summary += "- Production correlation <= 0.7 (hard constraint)\n"
        summary += f"- Max production correlation <= {self.max_corr_var.get()}\n\n"
        
        # Statistics
        if self.successful_alphas:
            avg_sharpe = sum(item['metrics'].sharpe for item in self.successful_alphas) / len(self.successful_alphas)
            avg_margin = sum(item['metrics'].margin for item in self.successful_alphas) / len(self.successful_alphas)
            avg_prod_corr = sum(item['correlation_analysis'].max_correlation for item in self.successful_alphas) / len(self.successful_alphas)
            
            summary += "Average metrics:\n"
            summary += f"- Average Sharpe: {avg_sharpe:.3f}\n"
            summary += f"- Average Margin: {avg_margin:.6f} ({avg_margin*10000:.1f} bps)\n"
            summary += f"- Average Production Correlation: {avg_prod_corr:.3f}\n\n"
            
            # Region distribution
            region_dist = {}
            for item in self.successful_alphas:
                region = item['metrics'].region
                region_dist[region] = region_dist.get(region, 0) + 1
            
            summary += "Region distribution:\n"
            for region, count in sorted(region_dist.items()):
                summary += f"- {region}: {count} alphas\n"
            summary += "\n"
            
            # Top performers
            summary += "Top 5 performers by Sharpe ratio:\n"
            sorted_alphas = sorted(self.successful_alphas, key=lambda x: x['metrics'].sharpe, reverse=True)
            for i, item in enumerate(sorted_alphas[:5], 1):
                metrics = item['metrics']
                corr_analysis = item['correlation_analysis']
                summary += f"{i}. {metrics.alpha_id}: Sharpe={metrics.sharpe:.3f}, "
                summary += f"Margin={metrics.margin:.6f}, ProdCorr={corr_analysis.max_correlation:.3f}\n"
        
        self.summary_text.insert(tk.END, summary)
    
    def _update_alphas_table(self):
        """Update the alphas table"""
        # Clear existing items
        for item in self.alphas_tree.get_children():
            self.alphas_tree.delete(item)
        
        # Add alphas to table
        for item in self.successful_alphas:
            metrics = item['metrics']
            corr_analysis = item['correlation_analysis']
            
            # Calculate self correlation (placeholder - would need actual self correlation data)
            self_corr = "N/A"  # This would need to be calculated from actual data
            
            values = (
                metrics.alpha_id,
                f"{metrics.sharpe:.3f}",
                f"{metrics.margin:.6f}",
                f"{corr_analysis.max_correlation:.3f}",
                self_corr,
                metrics.region,
                metrics.universe,
                metrics.status
            )
            
            self.alphas_tree.insert("", tk.END, values=values)
    
    def on_alpha_double_click(self, event):
        """Handle double-click on alpha row"""
        selection = self.alphas_tree.selection()
        if not selection:
            return
        
        item = self.alphas_tree.item(selection[0])
        alpha_id = item['values'][0]
        
        # Find the alpha data
        alpha_data = None
        for item in self.successful_alphas:
            if item['metrics'].alpha_id == alpha_id:
                alpha_data = item
                break
        
        if alpha_data:
            self.show_alpha_details(alpha_data)
    
    def show_alpha_details(self, alpha_data):
        """Show detailed information about an alpha"""
        metrics = alpha_data['metrics']
        corr_analysis = alpha_data['correlation_analysis']
        
        # Create details window
        details_window = tk.Toplevel(self.root)
        details_window.title(f"Alpha Details - {metrics.alpha_id}")
        details_window.geometry("800x600")
        
        # Create scrolled text widget
        details_text = scrolledtext.ScrolledText(details_window, wrap=tk.WORD)
        details_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Generate details
        details = f"Alpha Details: {metrics.alpha_id}\n"
        details += "=" * 50 + "\n\n"
        
        details += "Basic Information:\n"
        details += f"- Alpha ID: {metrics.alpha_id}\n"
        details += f"- Code: {metrics.code[:100]}{'...' if len(metrics.code) > 100 else ''}\n"
        details += f"- Region: {metrics.region}\n"
        details += f"- Universe: {metrics.universe}\n"
        details += f"- Delay: {metrics.delay}\n"
        details += f"- Neutralization: {metrics.neutralization}\n"
        details += f"- Date Created: {metrics.date_created}\n"
        details += f"- Status: {metrics.status}\n\n"
        
        details += "Performance Metrics:\n"
        details += f"- Sharpe Ratio: {metrics.sharpe:.3f}\n"
        details += f"- Fitness: {metrics.fitness:.3f}\n"
        details += f"- Returns: {metrics.returns:.3f}\n"
        details += f"- Turnover: {metrics.turnover:.3f}\n"
        details += f"- Drawdown: {metrics.drawdown:.3f}\n"
        details += f"- Margin: {metrics.margin:.6f} ({metrics.margin*10000:.1f} bps)\n"
        details += f"- PnL: {metrics.pnl:,}\n"
        details += f"- Long Count: {metrics.long_count}\n"
        details += f"- Short Count: {metrics.short_count}\n\n"
        
        details += "Check Results:\n"
        details += f"- Passed: {metrics.checks_passed}\n"
        details += f"- Failed: {metrics.checks_failed}\n"
        details += f"- Warning: {metrics.checks_warning}\n"
        details += f"- Pending: {metrics.checks_pending}\n\n"
        
        details += "Correlation Analysis:\n"
        details += f"- Max Production Correlation: {corr_analysis.max_correlation:.3f}\n"
        details += f"- Min Production Correlation: {corr_analysis.min_correlation:.3f}\n"
        details += f"- Risk Level: {corr_analysis.risk_level}\n"
        details += f"- High Correlation Count: {corr_analysis.high_correlation_count}\n"
        details += f"- Medium Correlation Count: {corr_analysis.medium_correlation_count}\n"
        details += f"- Low Correlation Count: {corr_analysis.low_correlation_count}\n"
        details += f"- Total Production Alphas: {corr_analysis.total_production_alphas}\n\n"
        
        if corr_analysis.recommendations:
            details += "Recommendations:\n"
            for rec in corr_analysis.recommendations:
                details += f"- {rec}\n"
        
        details_text.insert(tk.END, details)
        details_text.config(state=tk.DISABLED)
    
    def export_results(self):
        """Export results to JSON file"""
        if not self.successful_alphas:
            messagebox.showwarning("No Data", "No results to export")
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alpha_icu_gui_results_{timestamp}.json"
            
            # Prepare export data
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "criteria": {
                    "min_sharpe": 1.2,
                    "min_margin": 0.0008,
                    "max_prod_correlation": 0.7,
                    "max_corr_threshold": float(self.max_corr_var.get()),
                    "api_response_based": "FAIL=reject, WARNING=accept"
                },
                "summary": {
                    "total_alphas": len(self.successful_alphas),
                    "avg_sharpe": sum(item['metrics'].sharpe for item in self.successful_alphas) / len(self.successful_alphas),
                    "avg_margin": sum(item['metrics'].margin for item in self.successful_alphas) / len(self.successful_alphas),
                    "avg_prod_correlation": sum(item['correlation_analysis'].max_correlation for item in self.successful_alphas) / len(self.successful_alphas)
                },
                "alphas": []
            }
            
            # Add alpha data
            for item in self.successful_alphas:
                metrics = item['metrics']
                corr_analysis = item['correlation_analysis']
                
                alpha_data = {
                    "alpha_id": metrics.alpha_id,
                    "code": metrics.code,
                    "region": metrics.region,
                    "universe": metrics.universe,
                    "delay": metrics.delay,
                    "neutralization": metrics.neutralization,
                    "sharpe": metrics.sharpe,
                    "fitness": metrics.fitness,
                    "returns": metrics.returns,
                    "turnover": metrics.turnover,
                    "drawdown": metrics.drawdown,
                    "margin": metrics.margin,
                    "pnl": metrics.pnl,
                    "long_count": metrics.long_count,
                    "short_count": metrics.short_count,
                    "checks_passed": metrics.checks_passed,
                    "checks_failed": metrics.checks_failed,
                    "checks_warning": metrics.checks_warning,
                    "checks_pending": metrics.checks_pending,
                    "date_created": metrics.date_created,
                    "status": metrics.status,
                    "correlation_analysis": {
                        "max_correlation": corr_analysis.max_correlation,
                        "min_correlation": corr_analysis.min_correlation,
                        "risk_level": corr_analysis.risk_level,
                        "high_correlation_count": corr_analysis.high_correlation_count,
                        "medium_correlation_count": corr_analysis.medium_correlation_count,
                        "low_correlation_count": corr_analysis.low_correlation_count,
                        "total_production_alphas": corr_analysis.total_production_alphas,
                        "recommendations": corr_analysis.recommendations
                    }
                }
                export_data["alphas"].append(alpha_data)
            
            # Write to file
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            messagebox.showinfo("Export Complete", f"Results exported to {filename}")
            logger.info(f"Results exported to {filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")
            logger.error(f"Export failed: {e}")

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = AlphaICU_GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
