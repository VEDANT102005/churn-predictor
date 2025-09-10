import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import webbrowser
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

class ChurnPredictor:
    def __init__(self):
        self.model = LogisticRegression(random_state=42)
        self.encoders = {}
        self.feature_names = []
        
    def load_data(self, file_path=None):
        """Load any CSV file from data folder"""
        if file_path is None:
            data_folder = 'data/'
            if os.path.exists(data_folder):
                csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
                if csv_files:
                    file_path = os.path.join(data_folder, csv_files[0])
                    print(f"🔍 Found CSV file: {csv_files[0]}")
                else:
                    print("❌ No CSV files found in data/ folder")
                    return None
            else:
                print("❌ data/ folder not found")
                return None
        
        try:
            data = pd.read_csv(file_path)
            print(f"✅ Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
            return data, csv_files[0] if file_path else file_path
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None, None
    
    def preprocess_data(self, data):
        """Preprocess the data"""
        if 'Churn' not in data.columns:
            print("❌ Error: 'Churn' column not found in dataset")
            return None, None
        
        # Create a copy
        df = data.copy()
        
        # Handle the target variable
        if df['Churn'].dtype == 'object':
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        # Handle missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
            else:
                df[col] = df[col].fillna(df[col].median())
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if 'Churn' in categorical_cols:
            categorical_cols.remove('Churn')
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le
        
        # Prepare features and target
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        self.feature_names = X.columns.tolist()
        
        return X, y, df
    
    def train_model(self, X, y):
        """Train the logistic regression model"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        return accuracy, report, X_test.shape[0], cm, y_test, y_pred
    
    def get_feature_importance(self):
        """Get top 10 most important features"""
        coefficients = self.model.coef_[0]
        feature_importance = [(self.feature_names[i], abs(coefficients[i])) for i in range(len(coefficients))]
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        return feature_importance[:10]
    
    def create_visualizations(self, data, cm, feature_importance):
        """Create all visualizations and return as base64 encoded images"""
        plt.style.use('default')
        images = {}
        
        # 1. Churn Distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        churn_counts = data['Churn'].value_counts()
        colors = ['#43e97b', '#fa709a']
        bars = ax.bar(['No Churn', 'Churn'], churn_counts.values, color=colors, alpha=0.8)
        ax.set_title('Customer Churn Distribution', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Number of Customers', fontsize=12)
        
        # Add value labels on bars
        for bar, value in zip(bars, churn_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                   f'{value:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        images['churn_dist'] = self.fig_to_base64(fig)
        plt.close()
        
        # 2. Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'], 
                   yticklabels=['No Churn', 'Churn'],
                   ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        plt.tight_layout()
        images['confusion_matrix'] = self.fig_to_base64(fig)
        plt.close()
        
        # 3. Feature Importance
        fig, ax = plt.subplots(figsize=(10, 8))
        features, importances = zip(*feature_importance)
        y_pos = np.arange(len(features))
        
        bars = ax.barh(y_pos, importances, color='#667eea', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('Top 10 Most Important Features', fontsize=16, fontweight='bold', pad=20)
        ax.invert_yaxis()
        
        # Add value labels
        for bar, importance in zip(bars, importances):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{importance:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        images['feature_importance'] = self.fig_to_base64(fig)
        plt.close()
        
        # 4. Correlation Heatmap (top features only)
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = [f[0] for f in feature_importance] + ['Churn']
        corr_data = data[top_features].corr()
        
        mask = np.triu(np.ones_like(corr_data))
        sns.heatmap(corr_data, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Feature Correlation Matrix (Top Features)', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        images['correlation'] = self.fig_to_base64(fig)
        plt.close()
        
        # 5. Model Performance Metrics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy gauge
        accuracy_val = accuracy_score([1 if cm[1,1] > cm[0,0] else 0 for _ in range(cm.sum())], 
                                     [1 for _ in range(cm[1,1] + cm[0,1])] + [0 for _ in range(cm[0,0] + cm[1,0])])
        
        ax1.pie([accuracy_val, 1-accuracy_val], labels=['Correct', 'Incorrect'], 
               colors=['#43e97b', '#fa709a'], autopct='%1.1f%%', startangle=90)
        ax1.set_title('Model Accuracy', fontweight='bold')
        
        # Precision/Recall comparison
        metrics = ['Precision', 'Recall', 'F1-Score']
        no_churn = [cm[0,0]/(cm[0,0]+cm[1,0]), cm[0,0]/(cm[0,0]+cm[0,1]), 
                   2*(cm[0,0]/(cm[0,0]+cm[1,0]))*(cm[0,0]/(cm[0,0]+cm[0,1]))/((cm[0,0]/(cm[0,0]+cm[1,0]))+(cm[0,0]/(cm[0,0]+cm[0,1])))]
        churn = [cm[1,1]/(cm[1,1]+cm[0,1]), cm[1,1]/(cm[1,1]+cm[1,0]), 
                2*(cm[1,1]/(cm[1,1]+cm[0,1]))*(cm[1,1]/(cm[1,1]+cm[1,0]))/((cm[1,1]/(cm[1,1]+cm[0,1]))+(cm[1,1]/(cm[1,1]+cm[1,0])))]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax2.bar(x - width/2, no_churn, width, label='No Churn', color='#43e97b', alpha=0.8)
        ax2.bar(x + width/2, churn, width, label='Churn', color='#fa709a', alpha=0.8)
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Score')
        ax2.set_title('Performance Metrics Comparison', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        # Data quality overview
        missing_data = data.isnull().sum().head(10)
        if missing_data.sum() > 0:
            ax3.bar(range(len(missing_data)), missing_data.values, color='#667eea', alpha=0.8)
            ax3.set_title('Missing Values by Feature', fontweight='bold')
            ax3.set_xlabel('Features')
            ax3.set_ylabel('Missing Count')
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No Missing\nValues Found!', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=14, fontweight='bold', color='green')
            ax3.set_title('Data Quality Check', fontweight='bold')
        
        # Dataset overview
        ax4.text(0.1, 0.8, f'Total Records: {len(data):,}', transform=ax4.transAxes, fontsize=12, fontweight='bold')
        ax4.text(0.1, 0.6, f'Features: {len(data.columns)-1}', transform=ax4.transAxes, fontsize=12, fontweight='bold')
        ax4.text(0.1, 0.4, f'Churn Rate: {(data["Churn"].sum()/len(data)*100):.1f}%', transform=ax4.transAxes, fontsize=12, fontweight='bold')
        ax4.text(0.1, 0.2, f'Model: Logistic Regression', transform=ax4.transAxes, fontsize=12, fontweight='bold')
        ax4.set_title('Dataset Overview', fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        images['metrics_overview'] = self.fig_to_base64(fig)
        plt.close()
        
        return images
    
    def fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        return image_base64
    
    def generate_html_report(self, accuracy, report, test_size, feature_importance, filename, data_shape, images):
        """Generate HTML report with visualizations"""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Churn Prediction Results</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            padding: 40px 20px;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }}
        
        .stat-card.blue {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}
        
        .stat-card.green {{
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        }}
        
        .stat-card.purple {{
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }}
        
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            font-size: 1em;
            opacity: 0.9;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            color: #333;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .viz-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .viz-card {{
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            text-align: center;
        }}
        
        .viz-card h3 {{
            margin-bottom: 15px;
            color: #333;
            font-size: 1.3em;
        }}
        
        .viz-card img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
        }}
        
        .features-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }}
        
        .feature-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .feature-name {{
            font-weight: 600;
            color: #333;
        }}
        
        .feature-score {{
            background: #667eea;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.9em;
        }}
        
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }}
        
        .metrics-table th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        .metrics-table td {{
            padding: 15px;
            border-bottom: 1px solid #eee;
        }}
        
        .metrics-table tr:hover {{
            background: #f8f9fa;
        }}
        
        .timestamp {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            color: #666;
        }}
        
        .full-width {{
            grid-column: 1 / -1;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Customer Churn Prediction</h1>
            <p>Machine Learning Model Results with Interactive Visualizations</p>
        </div>
        
        <div class="content">
            <div class="stats-grid">
                <div class="stat-card blue">
                    <div class="stat-number">{accuracy:.1%}</div>
                    <div class="stat-label">🎯 Model Accuracy</div>
                </div>
                <div class="stat-card green">
                    <div class="stat-number">{data_shape[0]:,}</div>
                    <div class="stat-label">📋 Total Records</div>
                </div>
                <div class="stat-card purple">
                    <div class="stat-number">{test_size:,}</div>
                    <div class="stat-label">🧪 Test Samples</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{data_shape[1]-1}</div>
                    <div class="stat-label">🔢 Features Used</div>
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Data Visualizations</h2>
                <div class="viz-grid">
                    <div class="viz-card">
                        <h3>🥧 Churn Distribution</h3>
                        <img src="data:image/png;base64,{images['churn_dist']}" alt="Churn Distribution">
                    </div>
                    <div class="viz-card">
                        <h3>🎯 Confusion Matrix</h3>
                        <img src="data:image/png;base64,{images['confusion_matrix']}" alt="Confusion Matrix">
                    </div>
                </div>
                
                <div class="viz-grid">
                    <div class="viz-card full-width">
                        <h3>⭐ Feature Importance</h3>
                        <img src="data:image/png;base64,{images['feature_importance']}" alt="Feature Importance">
                    </div>
                </div>
                
                <div class="viz-grid">
                    <div class="viz-card">
                        <h3>🔥 Feature Correlations</h3>
                        <img src="data:image/png;base64,{images['correlation']}" alt="Correlation Matrix">
                    </div>
                    <div class="viz-card">
                        <h3>📊 Model Performance Overview</h3>
                        <img src="data:image/png;base64,{images['metrics_overview']}" alt="Metrics Overview">
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>🏆 Top 10 Most Important Features</h2>
                <div class="features-grid">
"""
        
        for i, (feature, importance) in enumerate(feature_importance, 1):
            html_content += f"""
                    <div class="feature-item">
                        <span class="feature-name">#{i} {feature}</span>
                        <span class="feature-score">{importance:.3f}</span>
                    </div>
"""
        
        html_content += f"""
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Detailed Performance Metrics</h2>
                <table class="metrics-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>No Churn (0)</th>
                            <th>Churn (1)</th>
                            <th>Overall</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Precision</strong></td>
                            <td>{report['0']['precision']:.3f}</td>
                            <td>{report['1']['precision']:.3f}</td>
                            <td>{report['macro avg']['precision']:.3f}</td>
                        </tr>
                        <tr>
                            <td><strong>Recall</strong></td>
                            <td>{report['0']['recall']:.3f}</td>
                            <td>{report['1']['recall']:.3f}</td>
                            <td>{report['macro avg']['recall']:.3f}</td>
                        </tr>
                        <tr>
                            <td><strong>F1-Score</strong></td>
                            <td>{report['0']['f1-score']:.3f}</td>
                            <td>{report['1']['f1-score']:.3f}</td>
                            <td>{report['macro avg']['f1-score']:.3f}</td>
                        </tr>
                        <tr>
                            <td><strong>Support</strong></td>
                            <td>{int(report['0']['support'])}</td>
                            <td>{int(report['1']['support'])}</td>
                            <td>{int(report['macro avg']['support'])}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>📊 Dataset Information</h2>
                <table class="metrics-table">
                    <tbody>
                        <tr>
                            <td><strong>📁 Dataset File</strong></td>
                            <td>{filename}</td>
                            <td></td>
                            <td></td>
                        </tr>
                        <tr>
                            <td><strong>📏 Dataset Shape</strong></td>
                            <td>{data_shape[0]:,} rows × {data_shape[1]} columns</td>
                            <td></td>
                            <td></td>
                        </tr>
                        <tr>
                            <td><strong>🎯 Model Type</strong></td>
                            <td>Logistic Regression</td>
                            <td></td>
                            <td></td>
                        </tr>
                        <tr>
                            <td><strong>✂️ Train/Test Split</strong></td>
                            <td>80% / 20%</td>
                            <td></td>
                            <td></td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div class="timestamp">
                <p>🕐 Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>🤖 Powered by Logistic Regression with Interactive Visualizations</p>
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        # Save HTML file
        html_filename = 'churn_prediction_results.html'
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_filename

def main():
    print("🚀 Customer Churn Prediction Model with Visualizations")
    print("=" * 60)
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Load data
    result = predictor.load_data()
    if result is None or result[0] is None:
        return
    
    data, filename = result
    
    # Preprocess data
    result = predictor.preprocess_data(data)
    if result is None or result[0] is None:
        return
    
    X, y, processed_data = result
    print(f"⚙️  Preprocessing completed")
    
    # Train model
    print(f"🤖 Training Logistic Regression model...")
    accuracy, report, test_size, cm, y_test, y_pred = predictor.train_model(X, y)
    
    # Get feature importance
    feature_importance = predictor.get_feature_importance()
    
    # Create visualizations
    print(f"🎨 Generating visualizations...")
    images = predictor.create_visualizations(processed_data, cm, feature_importance)
    
    # Generate HTML report
    print(f"📄 Generating HTML report with visualizations...")
    html_file = predictor.generate_html_report(
        accuracy, report, test_size, feature_importance, filename, data.shape, images
    )
    
    # Open in browser
    print(f"🌐 Opening interactive results in browser...")
    webbrowser.open(html_file)
    
    print(f"✅ Complete! Interactive results saved as '{html_file}'")
    print(f"🎯 Model Accuracy: {accuracy:.1%}")
    print(f"📊 Check your browser for detailed visualizations and analysis!")

if __name__ == "__main__":
    main()