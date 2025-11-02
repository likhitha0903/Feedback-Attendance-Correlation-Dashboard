import os
import base64
from io import BytesIO
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import json
import uuid
from datetime import datetime
plt.switch_backend('Agg')

def get_datasets():
    """Get list of all available datasets"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_file = os.path.join(base_dir, "data", "datasets.json")
    
    # Default dataset structure
    default_datasets = {
        "default": {
            "name": "Default Dataset",
            "filename": "data.csv", 
            "path": "default/data.csv",
            "upload_date": "2024-01-01",
            "rows": 0,
            "columns": 0,
            "is_default": True
        }
    }
    
    if os.path.exists(datasets_file):
        try:
            with open(datasets_file, 'r') as f:
                return json.load(f)
        except:
            return default_datasets
    
    return default_datasets

def save_dataset_info(datasets):
    """Save dataset metadata"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_file = os.path.join(base_dir, "data", "datasets.json")
    with open(datasets_file, 'w') as f:
        json.dump(datasets, f, indent=2)

def add_uploaded_dataset(file, filename):
    """Add new uploaded dataset"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate unique ID
    dataset_id = str(uuid.uuid4())[:8]
    new_filename = f"dataset_{dataset_id}.csv"
    
    # Save file
    upload_dir = os.path.join(base_dir, "data", "uploaded")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, new_filename)
    file.save(file_path)
    
    # Get dataset stats
    df = pd.read_csv(file_path)
    
    # Update datasets metadata
    datasets = get_datasets()
    datasets[dataset_id] = {
        "name": filename,
        "filename": new_filename,
        "path": f"uploaded/{new_filename}",
        "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "rows": len(df),
        "columns": len(df.columns),
        "is_default": False
    }
    
    save_dataset_info(datasets)
    return dataset_id

def load_data(dataset_id="default"):
    """Load data from specific dataset"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        datasets = get_datasets()
        
        if dataset_id not in datasets:
            dataset_id = "default"  # Fallback to default
        
        dataset_info = datasets[dataset_id]
        data_path = os.path.join(base_dir, "data", dataset_info["path"])
        
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()
        
        # Update row/column count
        dataset_info["rows"] = len(df)
        dataset_info["columns"] = len(df.columns)
        
        return df, dataset_info
    except Exception as e:
        print(f"❌ Error loading dataset {dataset_id}: {e}")
        # Fallback to default dataset
        if dataset_id != "default":
            return load_data("default")
        else:
            raise e
            
def sunburst_chart(df):
    required_cols = ["Department", "Teacher_Feedback", "Average_Attendance_On_Class"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing in the DataFrame.")

    df_clean = df.dropna(subset=required_cols)

    if df_clean.empty:
        return "<p>No data available for Sunburst chart.</p>"

    fig = px.sunburst(
        df_clean,
        path=['Department', 'Teacher_Feedback'],
        values='Average_Attendance_On_Class',
        title="Department vs Teacher Feedback (by Average Attendance)",
        height=600,
        width=800
    )

    return fig.to_html(full_html=False)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
import base64
def univariate(df, numeric_discrete_cols=None):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    import plotly.express as px

    numeric_discrete_cols = numeric_discrete_cols or []
    column_plots = {}

    # 🧹 Drop unwanted columns
    df = df.drop(columns=[col for col in ['Student_ID', 'password', 'Name'] if col in df.columns])

    for col in df.columns:
        # 🌿 NUMERIC
        if pd.api.types.is_numeric_dtype(df[col]):
            plt.figure(figsize=(5.5, 3))
            sns.histplot(df[col].dropna(), kde=True, bins=20,
                         color="#6fa3ef", edgecolor="white", linewidth=1.2)
            plt.title(f"{col.replace('_', ' ').title()}", fontsize=13, weight="bold", color="#2e86c1")
            plt.xlabel("")
            plt.ylabel("")
            plt.grid(alpha=0.2)
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png", bbox_inches='tight', dpi=100)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            plt.close()

            # ✨ Full descriptive stats
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
            stats = {
                "Count": df[col].count(),
                "Missing": df[col].isna().sum(),
                "Mean": f"{df[col].mean():.2f}",
                "Median": f"{df[col].median():.2f}",
                "Mode": mode_val,
                "Min": f"{df[col].min():.2f}",
                "Max": f"{df[col].max():.2f}",
                "Range": f"{df[col].max() - df[col].min():.2f}",
                "Std Dev": f"{df[col].std():.2f}",
                "Variance": f"{df[col].var():.2f}",
                "Skewness": f"{df[col].skew():.2f}",
                "Kurtosis": f"{df[col].kurt():.2f}"
            }

            # 🩵 Enhanced numeric stats card
            stats_html = f"""
            <div style="
                background: linear-gradient(145deg, #f9fcff, #edf4ff);
                border-radius: 18px;
                padding: 28px 24px;
                margin-bottom: 30px;
                box-shadow: 0 6px 18px rgba(46,134,193,0.1);
                transition: 0.3s ease-in-out;
            ">
                <h3 style='color:#2e86c1; font-weight:600; font-size:1.2rem; margin-bottom:18px;'>
                    📈 {col.replace('_', ' ').title()}
                </h3>
                <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr)); gap:10px; font-size:0.93rem; color:#333;">
                    {''.join(f"<div><b>{k}:</b> {v}</div>" for k,v in stats.items())}
                </div>
                <img src='data:image/png;base64,{img_base64}' 
                     style='width:100%; margin-top:18px; border-radius:12px; box-shadow:0 2px 12px rgba(0,0,0,0.05);'/>
            </div>
            """
            column_plots[col] = stats_html

        # 🍵 CATEGORICAL — matcha mood
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            cats = df[col].value_counts(dropna=False)
            top_5 = cats.nlargest(5)
            top_cat = top_5.index[0]
            top_pct = (top_5.iloc[0] / len(df)) * 100

            fig = px.bar(
                x=top_5.index.astype(str),
                y=top_5.values,
                text=[f"{v}" for v in top_5.values],
                color=top_5.values,
                color_continuous_scale=["#e3fcec", "#b7e4c7", "#95d5b2", "#74c69d", "#52b788"],
                title=None,
                width=500, height=340
            )
            fig.update_traces(
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>',
                marker_line_color='rgba(255,255,255,0.6)',
                marker_line_width=1.3
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title="",
                yaxis_title="",
                margin=dict(t=20, l=10, r=10, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#2e7d32")
            )

            stats_html = f"""
            <div style="
                background: linear-gradient(145deg, #f5fff8, #ebf9f0);
                border-radius: 18px;
                padding: 28px 24px;
                margin-bottom: 30px;
                box-shadow: 0 6px 18px rgba(46,134,85,0.1);
                transition: 0.3s ease-in-out;
            ">
                <h3 style='color:#1b5e20; font-weight:600; font-size:1.2rem; margin-bottom:18px;'>
                    📊 {col.replace('_', ' ').title()}
                </h3>
                <div style='font-size:0.95rem; line-height:1.6; color:#333;'>
                    <b>Unique Categories:</b> {df[col].nunique()} <br>
                    <b>Top Category:</b> {top_cat} ({top_pct:.1f}%) <br>
                    <b>Missing:</b> {df[col].isna().sum()}
                </div>
                <div style='margin-top:15px;'>{fig.to_html(full_html=False, include_plotlyjs="cdn")}</div>
            </div>
            """
            column_plots[col] = stats_html

    return column_plots


def generate_insights():
    """
    Returns a dictionary with HTML-ready insights summary.
    Designed for Flask dashboards (e.g., to display after data analysis).
    """
    insights = [
        "There are more female students than male students.",
        "AI-DS has the largest student strength among all departments.",
        "The average attendance maintained by each department is 73%.",
        "A few departments are actively participating in co-curricular activities.",
        "Each department maintains at least a 7.25 CGPA.",
        "The average teacher feedback across all departments is 7.0.",
        "43.1% of students received a ‘Good’ feedback rating from teachers."
    ]

    insights_html = """
    <div style='background:#f8f9fa; border-radius:10px; padding:15px; 
                box-shadow:0 0 10px rgba(0,0,0,0.1); margin-top:20px;'>
        <h3 style='color:#154360; text-align:center;'>📊 Key Insights</h3>
        <ul style='line-height:1.8; color:#2e4053; font-size:15px;'>
    """
    for point in insights:
        insights_html += f"<li>✅ {point}</li>"
    insights_html += "</ul></div>"

    return {"title": "Key Insights", "content": insights_html}

def gender_vs_department_all(df):
    """
    Generates multiple bi-variate visualizations for Gender vs Department
    Returns a list of dictionaries with 'title', 'plot', and 'note'
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    from io import BytesIO
    import base64
    import pandas as pd

    plots = []
    if 'Gender' not in df.columns or 'Department' not in df.columns:
        return [{"title": "Error", "plot": None, "note": "Missing 'Gender' or 'Department' columns"}]

    # Clean data
    df = df.dropna(subset=['Gender', 'Department'])
    df = df[(df['Gender'].astype(str).str.strip() != "") & (df['Department'].astype(str).str.strip() != "")]

    # ------------------ 1️⃣ BASE COUNTS ------------------
    counts = df.groupby(['Department', 'Gender']).size().unstack(fill_value=0)


    # ------------------ 3️⃣ PREP LONG-FORM DATA FOR PLOTLY ------------------
    df_counts = counts.reset_index().melt(id_vars='Department', var_name='Gender', value_name='Gender Count')

    # ------------------ 4️⃣ INTERACTIVE GROUPED BAR ------------------
    fig = px.bar(
        df_counts,
        x='Department',
        y='Gender Count',
        color='Gender',
        barmode='group',
        title="Interactive Grouped Bar"
    )
    fig.update_traces(
        hovertemplate='<b>Department:</b> %{x}<br><b>Gender:</b> %{customdata[0]}<br><b>Count:</b> %{y}',
        customdata=df_counts[['Gender']].values
    )
    plots.append({
        "title": "Interactive Grouped Bar (Plotly)",
        "plot": fig.to_html(full_html=False),
        "note": "Hover-friendly grouped bars for each department with gender counts."
    })

    # ------------------ 5️⃣ INTERACTIVE STACKED BAR ------------------
    fig2 = px.bar(
        df_counts,
        x='Department',
        y='Gender Count',
        color='Gender',
        barmode='stack',
        title="Interactive Stacked Bar"
    )
    fig2.update_traces(
        hovertemplate='<b>%{x}</b><br>Gender: %{customdata[0]}<br>Count: %{y}',
        customdata=df_counts[['Gender']].values
    )
    plots.append({
        "title": "Interactive Stacked Bar (Plotly)",
        "plot": fig2.to_html(full_html=False),
        "note": "Hover-friendly stacked view for departments with gender counts."
    })

     # ------------------ 8️⃣ HEATMAP ------------------
    fig6 = px.imshow(
        counts,
        text_auto=True,
        color_continuous_scale="YlGnBu",
        title="Heatmap – Gender Count per Department"
    )
    fig6.update_layout(
        xaxis_title="Gender",
        yaxis_title="Department",
        coloraxis_showscale=True
    )
    plots.append({
        "title": "Interactive Heatmap",
        "plot": fig6.to_html(full_html=False, include_plotlyjs='cdn'),
        "note": "Interactive view of gender counts per department."
    })

    # ------------------ 6️⃣ SUNBURST ------------------
    fig3 = px.sunburst(df, path=['Department', 'Gender'], title="Sunburst: Department → Gender")
    plots.append({
        "title": "Sunburst Chart",
        "plot": fig3.to_html(full_html=False),
        "note": "Shows department-to-gender hierarchy visually."
    })
    # ✅ Step 1: Select top 10 students per department by CGPA
    top10_per_dept = (
        df.sort_values(['Department', 'What_Is_Your_Current_CGPA?'], ascending=[True, False])
        .groupby('Department')
        .head(10)
        .reset_index(drop=True)
        [['Department', 'Gender', 'Student_ID', 'What_Is_Your_Current_CGPA?']]
    )

    # ✅ Step 2: Create Treemap with Gender as color + custom hover data
    fig5 = px.treemap(
        top10_per_dept,
        path=['Department', 'Student_ID'],               # Hierarchy: Department → Student
        values='What_Is_Your_Current_CGPA?',             # Size = CGPA
        color='Gender',                                  # Color by Gender
        color_discrete_sequence=px.colors.qualitative.Set3,  # Balanced neutral palette
        custom_data=['Gender', 'What_Is_Your_Current_CGPA?'], # so we can show labels properly
        title="Top 10 Students per Department (Gender-Colored, Size = CGPA)"
    )

    # ✅ Step 3: Custom hover template (use custom_data instead of %{color})
    fig5.update_traces(
        hovertemplate=(
            '<b>Student ID:</b> %{label}<br>'
            '<b>Department:</b> %{parent}<br>'
            '<b>Gender:</b> %{customdata[0]}<br>'
            '<b>CGPA:</b> %{customdata[1]}<extra></extra>'
        )
    )

    fig5.update_layout(
        legend_title_text="Gender",
        coloraxis_showscale=False,
        margin=dict(t=60, l=30, r=30, b=30)
    )

    # ✅ Step 4: Append to Flask plots
    plots.append({
        "title": "Top 10 Students Treemap",
        "plot": fig5.to_html(full_html=False),
        "note": "Displays top 10 students per department with gender-based colors and hover details."
    })



    return plots

def generate_insights_Academic():
    """
    Returns a dictionary with HTML-ready insights summary.
    Designed for Flask dashboards (e.g., to display after data analysis).
    """
    insights = [
        "Male students show a slightly higher average CGPA with greater variability, while females are more consistent overall.",
        "Mechanical tops in performance (avg 7.42); AI-DS trails slightly (avg 6.92), with all departments peaking near 9.9 CGPA."
    ]

    insights_html = """
    <div style='background:#f8f9fa; border-radius:10px; padding:15px; 
                box-shadow:0 0 10px rgba(0,0,0,0.1); margin-top:20px;'>
        <h3 style='color:#154360; text-align:center;'>📊 Key Insights</h3>
        <ul style='line-height:1.8; color:#2e4053; font-size:15px;'>
    """
    for point in insights:
        insights_html += f"<li>✅ {point}</li>"
    insights_html += "</ul></div>"

    return {"title": "Key Insights", "content": insights_html}


def generate_insights_Attendance(): 
    """
    Returns a dictionary with HTML-ready insights summary.
    Designed for Flask dashboards (e.g., to display after data analysis).
    """
    insights = [
        "Males have a slightly higher average and median attendance, though females show a more balanced, less extreme pattern overall.",
        "AI-DS and Mechanical lead with higher attendance, while IT and CSE lag slightly, showing more irregular participation."
    ]

    insights_html = """
    <div style='background:#f8f9fa; border-radius:10px; padding:15px; 
                box-shadow:0 0 10px rgba(0,0,0,0.1); margin-top:20px;'>
        <h3 style='color:#154360; text-align:center;'>📊 Key Insights</h3>
        <ul style='line-height:1.8; color:#2e4053; font-size:15px;'>
    """
    for point in insights:
        insights_html += f"<li>✅ {point}</li>"
    insights_html += "</ul></div>"

    return {"title": "Key Insights", "content": insights_html}

def generate_insights_Feedback(): 
    """
    Returns a dictionary with HTML-ready insights summary.
    Designed for Flask dashboards (e.g., to display after data analysis).
    """
    insights = [
        "Feedback ratings are nearly identical across genders — males (avg 6.9) slightly ahead, but overall spread and medians (~7) remain consistent.",
        "Mechanical leads in satisfaction (avg 7.05), while CSE trails slightly (avg 6.61); all departments show balanced distributions with most ratings between 6–8."
    ]

    insights_html = """
    <div style='background:#f8f9fa; border-radius:10px; padding:15px; 
                box-shadow:0 0 10px rgba(0,0,0,0.1); margin-top:20px;'>
        <h3 style='color:#154360; text-align:center;'>📊 Key Insights</h3>
        <ul style='line-height:1.8; color:#2e4053; font-size:15px;'>
    """
    for point in insights:
        insights_html += f"<li>✅ {point}</li>"
    insights_html += "</ul></div>"

    return {"title": "Key Insights", "content": insights_html}

def generate_insights_Engagement(): 
    """
    Returns a dictionary with HTML-ready insights summary.
    Designed for Flask dashboards (e.g., to display after data analysis).
    """
    insights = [
        "Gender vs Co-Curricular: Both genders show near-equal participation, with females (95 vs 87) leaning slightly more toward non-involvement.",
        "Department vs Co-Curricular: AI–DS and IT lead in participation, while ECE lags behind; others remain fairly balanced.",
        "Co-Curricular vs Attendance: Non-participants maintain slightly higher and steadier attendance (≈74%) than participants (≈72%).",
        "Co-Curricular vs CGPA: Participants outperform non-participants (7.30 vs 6.99), suggesting engagement enhances academic performance."
    ]

    insights_html = """
    <div style='background:#f8f9fa; border-radius:10px; padding:15px; 
                box-shadow:0 0 10px rgba(0,0,0,0.1); margin-top:20px;'>
        <h3 style='color:#154360; text-align:center;'>📊 Key Insights</h3>
        <ul style='line-height:1.8; color:#2e4053; font-size:15px;'>
    """
    for point in insights:
        insights_html += f"<li>✅ {point}</li>"
    insights_html += "</ul></div>"

    return {"title": "Key Insights", "content": insights_html}


def generate_insights_Correlation(): 
    """
    Returns a dictionary with HTML-ready insights summary.
    Designed for Flask dashboards (e.g., to display after data analysis).
    """
    insights = [
        # Newly added correlations and scatter insights
        "📈 Feedback vs CGPA:<br>There’s a strong positive link — students who receive higher teacher feedback ratings tend to have higher CGPAs. Correlation is very strong (r = 0.84).",

"📊 Attendance vs CGPA:<br>A moderate positive relationship exists (r = 0.50). Students with better attendance generally achieve higher CGPAs, but other factors also play a role.",

"📚 Attendance vs Feedback:<br>Teachers tend to rate students with good attendance more positively (r = 0.73), showing that consistent presence in class boosts teacher perception and overall performance." ]

    insights_html = """
    <div style='background:#f8f9fa; border-radius:10px; padding:15px; 
                box-shadow:0 0 10px rgba(0,0,0,0.1); margin-top:20px;'>
        <h3 style='color:#154360; text-align:center;'>📊 Key Insights</h3>
        <ul style='line-height:1.8; color:#2e4053; font-size:15px;'>
    """
    for point in insights:
        insights_html += f"<li>{point}</li>"
    insights_html += "</ul></div>"

    return {"title": "Key Insights", "content": insights_html}


def gender_vs_avg_attendance_all(df):
    """
    Generates multiple bi-variate visualizations for Gender vs Average Attendance (continuous)
    Returns a list of dictionaries with 'title', 'plot', and 'note'
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    from io import BytesIO
    import base64
    import pandas as pd

    plots = []

    # Ensure required columns exist
    if 'Gender' not in df.columns or 'Average_Attendance_On_Class' not in df.columns:
        return [{"title": "Error", "plot": None, "note": "Missing 'Gender' or 'Average_Attendance_On_Class' columns"}]

    # Clean data
    df = df.dropna(subset=['Gender', 'Average_Attendance_On_Class'])
    df = df[(df['Gender'].astype(str).str.strip() != "") & (pd.to_numeric(df['Average_Attendance_On_Class'], errors='coerce').notnull())]
    df['Average_Attendance_On_Class'] = pd.to_numeric(df['Average_Attendance_On_Class'], errors='coerce')
     # ------------------ 9️⃣ PLOTLY BAR MEAN ------------------
    mean_df = df.groupby('Gender')['Average_Attendance_On_Class'].mean().reset_index()
    fig3 = px.bar(mean_df, x='Gender', y='Average_Attendance_On_Class', color='Gender', title="Gender vs Attendance - Mean bar")
    plots.append({
        "title": "Interactive Mean Bar",
        "plot": fig3.to_html(full_html=False),
        "note": "Simplified mean comparison with hoverable values."
    })
     # ------------------ 7️⃣ PLOTLY INTERACTIVE BOX ------------------
    fig = px.box(df, x='Gender', y='Average_Attendance_On_Class', color='Gender', title="Gender vs Attendance - Box Plot")
    plots.append({
        "title": "Gender vs Attendance - Box plot",
        "plot": fig.to_html(full_html=False),
        "note": "Hover for stats like median, IQR, and outliers."
    })
   
    # ------------------ 5️⃣ KDE / DENSITY PLOT ------------------
    plt.figure(figsize=(6, 4))

    # Ensure numeric values (avoids any string conversion issues)
    df['Average_Attendance_On_Class'] = pd.to_numeric(df['Average_Attendance_On_Class'], errors='coerce')
    df = df.dropna(subset=['Average_Attendance_On_Class', 'Gender'])

    # KDE plot by gender
    for gender in df['Gender'].unique():
        sns.kdeplot(
            df[df['Gender'] == gender]['Average_Attendance_On_Class'],
            label=gender,
            fill=True
        )

    plt.title("Gender vs Avg Attendance – KDE Density Plot")
    plt.xlabel("Average Attendance (%)")
    plt.legend()

    # Save plot to buffer
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    encoded_img = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    # 🧩 Embed directly in HTML
    plots.append({
        "title": "Density Plot (KDE)",
        "plot": f"<img src='data:image/png;base64,{encoded_img}' alt='Density Plot (KDE)' class='img-fluid rounded shadow'>",
        "note": "Compares overall attendance distribution across genders."
    })

   

    return plots

def gender_vs_cocurricular_all(df):
    """
    Generates multiple bi-variate visualizations for Gender vs Are_You_Engaged_With_Any_Co-Curriculum_Activities? (Yes/No)
    Returns a list of dictionaries with 'title', 'plot', and 'note'
    """

    import plotly.express as px
    import pandas as pd

    plots = []

    # ✅ Ensure required columns exist
    col1, col2 = 'Gender', 'Are_You_Engaged_With_Any_Co-Curriculum_Activities?'
    if col1 not in df.columns or col2 not in df.columns:
        return [{"title": "Error", "plot": None, "note": f"Missing '{col1}' or '{col2}' columns"}]

    # ✅ Clean data
    df = df.dropna(subset=[col1, col2])
    df = df[(df[col1].astype(str).str.strip() != "") & (df[col2].astype(str).str.strip() != "")]
    df[col2] = df[col2].str.strip().str.title()

    # ✅ Count and Percent data
    counts = df.groupby([col1, col2]).size().reset_index(name="Count")
    total_counts = df.groupby(col1).size().reset_index(name="Total")
    percent = pd.merge(counts, total_counts, on=col1)
    percent["Percentage"] = (percent["Count"] / percent["Total"]) * 100

    # 🎨 Color scheme
    color_map = {"Yes": "#2ecc71", "No": "#e74c3c"}  # Green for Yes, Red for No


    # ------------------- 2️⃣ INTERACTIVE GROUPED BAR -------------------
    fig2 = px.bar(
        counts,
        x=col1,
        y="Count",
        color=col2,
        barmode="group",
        color_discrete_map=color_map,
        text="Count",
        hover_data={"Count": True, col1: True, col2: True},
        title="Grouped Bar – Gender vs Co-Curricular Participation"
    )
    fig2.update_layout(
        template="plotly_white",
        height=500,
        xaxis_title="Gender",
        yaxis_title="Number of Students",
        legend_title="Co-Curricular Participation",
        title_x=0.5,
        margin=dict(t=80, b=60, l=60, r=40),
    )
    fig2.update_traces(texttemplate="%{text}", textposition="outside")
    plots.append({
        "title": "Interactive Grouped Bar (Plotly)",
        "plot": fig2.to_html(full_html=False),
        "note": "Compares participation side-by-side for each gender."
    })

    # ------------------- 3️⃣ INTERACTIVE 100% STACKED BAR -------------------

    return plots

def gender_vs_cgpa(df):
    """
    Returns a list of dicts (each with title, plot, and note)
    for different Gender vs What_Is_Your_Current_CGPA? visualizations.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    from io import BytesIO
    import base64

    plots = []

    # ✅ Check for required columns
    if 'Gender' not in df.columns or 'What_Is_Your_Current_CGPA?' not in df.columns:
        return [{"title": "Error", "plot": None, "note": "Missing required columns: Gender or CGPA"}]

    # ---------- 3️⃣ Bar Plot (Average CGPA)
   # ---------- 3️⃣ Interactive Bar Plot (Average CGPA)
    avg_cgpa = df.groupby("Gender")["What_Is_Your_Current_CGPA?"].mean().reset_index()

    fig = px.bar(
        avg_cgpa,
        x="Gender",
        y="What_Is_Your_Current_CGPA?",
        color="Gender",
        text="What_Is_Your_Current_CGPA?",
        title="Gender vs CGPA — Interactive Average Bar Plot",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        autosize=True,
        height=None,
        width=None,
        margin=dict(l=20, r=20, t=70, b=50),
    )


    plots.append({
        "title": "Interactive Average Bar Plot",
        "plot": fig.to_html(full_html=False),
        "note": "Displays mean CGPA for each gender interactively with hover info."
    })

    # ---------- 5️⃣ Interactive Box Plot (Plotly)
    fig = px.box(df, x='Gender', y='What_Is_Your_Current_CGPA?', color='Gender',
                 title='Gender vs CGPA — Interactive Box Plot')
    plots.append({
        "title": "Interactive Box Plot",
        "plot": fig.to_html(full_html=False),
        "note": "Hover to view median, quartiles, and outliers."
    })
    return plots

def dept_vs_avg_attendance(df):
    """
    Returns a list of dicts (each containing title, plot, and note)
    for Department vs Average_Attendance_On_Class visualizations.
    Includes static (matplotlib) + interactive (plotly) plots.
    """
    import matplotlib
    matplotlib.use("Agg")  # ✅ Non-GUI backend for Flask
    import matplotlib.pyplot as plt
    import seaborn as sns
    from io import BytesIO
    import base64
    import plotly.express as px
    import pandas as pd

    plots = []

    # 🧩 Safety check
    if 'Department' not in df.columns or 'Average_Attendance_On_Class' not in df.columns:
        return [{"title": "Error", "plot": None, "note": "Missing required columns"}]

    # 💡 Sort departments by mean attendance for consistent x-axis
    order = (
        df.groupby("Department")["Average_Attendance_On_Class"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
     # ---------- 4️⃣ Interactive Bar Plot (Plotly)
    mean_df = df.groupby('Department', as_index=False)['Average_Attendance_On_Class'].mean()
    fig = px.bar(
        mean_df,
        x='Department',
        y='Average_Attendance_On_Class',
        color='Department',
        category_orders={'Department': order},
        text_auto='.2f',
        title="Average Attendance by Department - Bar Plot ",
        labels={"Average_Attendance_On_Class": "Average Attendance (%)"}
    )
    fig.update_traces(textposition='outside')
    plots.append({
        "title": "Interactive Bar Plot (Plotly)",
        "plot": fig.to_html(full_html=False),
        "note": "Displays mean attendance interactively per department with labels."
    })

    # ---------- 3️⃣ Interactive Box Plot (Plotly)
    fig = px.box(
        df,
        x='Department',
        y='Average_Attendance_On_Class',
        color='Department',
        category_orders={'Department': order},
        points='outliers',
        title="Department vs Avg Attendance - Box Plot",
        labels={"Average_Attendance_On_Class": "Average Attendance (%)"}
    )
    plots.append({
        "title": "Interactive Box Plot (Plotly)",
        "plot": fig.to_html(full_html=False),
        "note": "Hover to explore department-wise attendance ranges and outliers."
    })

   

    return plots

def dept_vs_cgpa_all(df):
    """
    Generates multiple visualizations for Department vs CGPA.
    Returns a list of dicts: {title, plot, note}, Flask dashboard compatible.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    from io import BytesIO
    import base64

    plots = []

    # --- Column check ---
    if 'Department' not in df.columns or 'What_Is_Your_Current_CGPA?' not in df.columns:
        return [{"title": "Error", "plot": None, "note": "Missing required columns"}]

    # --- Clean data ---
    df = df.dropna(subset=['Department', 'What_Is_Your_Current_CGPA?'])
    df = df[df['Department'].astype(str).str.strip() != ""]
    df['What_Is_Your_Current_CGPA?'] = pd.to_numeric(df['What_Is_Your_Current_CGPA?'], errors='coerce')
    df = df.dropna(subset=['What_Is_Your_Current_CGPA?'])

    # --- Sort departments by mean CGPA ---
    order = df.groupby("Department")["What_Is_Your_Current_CGPA?"].mean().sort_values().index


    # 1️⃣ Static Bar Plot (Mean CGPA per Department)
    mean_df = df.groupby('Department')['What_Is_Your_Current_CGPA?'].mean().reset_index()

    fig = px.bar(mean_df, x='Department', y='What_Is_Your_Current_CGPA?', color='Department',
                 text_auto='.2f', category_orders={'Department': list(order)},
                 title="Average CGPA by Department -Interactive Bar")
    plots.append({
        "title": "Interactive Bar (Plotly)",
        "plot": fig.to_html(full_html=False),
        "note": "Hover for mean CGPA per department with color-coded comparison."
    })
   

    # 3️⃣ Interactive Box Plot
    fig = px.box(df, x='Department', y='What_Is_Your_Current_CGPA?', color='Department',
                 points='outliers', category_orders={'Department': list(order)},
                 title="Department vs CGPA - Interactive Box Plot")
    plots.append({
        "title": "Interactive Box Plot (Plotly)",
        "plot": fig.to_html(full_html=False),
        "note": "Hover to explore CGPA ranges, medians, and outliers per department."
    })

    # 5️⃣ (Optional) Interactive Bar Plot (Mean CGPA)
   

    return plots

def dept_vs_feedback_all(df):
    """
    Generates Department vs Feedback Rating visualizations.
    Returns Flask-compatible list of dicts: {title, plot, note}.
    Includes static (matplotlib) + interactive (Plotly) charts.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    from io import BytesIO
    import base64

    plots = []

    # --- Column validation ---
    if 'Department' not in df.columns or 'Feedback_Rating' not in df.columns:
        return [{"title": "Error", "plot": None, "note": "Missing required columns"}]

    # --- Clean and prep data ---
    df = df.dropna(subset=['Department', 'Feedback_Rating'])
    df = df[df['Department'].astype(str).str.strip() != ""]
    df['Feedback_Rating'] = pd.to_numeric(df['Feedback_Rating'], errors='coerce')
    df = df.dropna(subset=['Feedback_Rating'])

    # --- Sort departments by mean rating (for consistent order across visuals) ---
    order = df.groupby("Department")["Feedback_Rating"].mean().sort_values().index


    # 1️⃣ Bar Plot — Mean Feedback per Department
    mean_df = df.groupby('Department')['Feedback_Rating'].mean().reset_index()
    fig = px.bar(mean_df, x='Department', y='Feedback_Rating', color='Department',
                 text_auto='.2f', category_orders={'Department': list(order)},
                 title="Interactive Bar — Average Feedback Rating by Department")
    plots.append({
        "title": "Interactive Bar (Plotly)",
        "plot": fig.to_html(full_html=False),
        "note": "Hover for mean feedback rating by department, color-coded for comparison."
    })

   

    # 3️⃣ Interactive Box Plot — Department vs Feedback Rating
    fig = px.box(df, x='Department', y='Feedback_Rating', color='Department',
                 points='outliers', category_orders={'Department': list(order)},
                 title="Interactive Box Plot — Department vs Feedback Rating")
    plots.append({
        "title": "Interactive Box Plot (Plotly)",
        "plot": fig.to_html(full_html=False),
        "note": "Hover to explore feedback rating ranges, medians, and outliers."
    })


    # 5️⃣ Optional: Interactive Bar Plot — Mean Feedback Rating
    
    return plots

def dept_vs_teacher_feedback(df):
    """
    Returns a list of dicts for Department vs Teacher Feedback (categorical)
    compatible with Flask dashboard template.
    """
    import pandas as pd
    import plotly.express as px

    plots = []

    if 'Department' not in df.columns or 'Teacher_Feedback' not in df.columns:
        return [{"title": "Error", "plot": None, "note": "Missing required columns"}]

    # 1️⃣ Stacked Bar Chart
    fig = px.histogram(
        df, x="Department", color="Teacher_Feedback", barmode="stack",
        title="Department vs Teacher Feedback — Stacked Bar",
        labels={"Teacher_Feedback": "Feedback Category"}
    )
    plots.append({
        "title": "Stacked Bar Chart (Plotly)",
        "plot": fig.to_html(full_html=False),
        "note": "Shows proportions of feedback within each department."
    })

    # 2️⃣ Grouped Bar Chart
    fig = px.histogram(
        df, x="Department", color="Teacher_Feedback", barmode="group",
        title="Department vs Teacher Feedback — Grouped Bar"
    )
    plots.append({
        "title": "Grouped Bar Chart (Plotly)",
        "plot": fig.to_html(full_html=False),
        "note": "Compares feedback categories side-by-side."
    })

    # 3️⃣ Heatmap
    pivot = df.pivot_table(index="Department", columns="Teacher_Feedback", aggfunc="size", fill_value=0)
    fig = px.imshow(
        pivot, text_auto=True, color_continuous_scale="Blues",
        title="Heatmap — Teacher Feedback per Department",
        labels=dict(x="Teacher Feedback", y="Department", color="Count")
    )
    plots.append({
        "title": "Heatmap (Plotly)",
        "plot": fig.to_html(full_html=False),
        "note": "Shows distribution of feedback intensity per department."
    })

    return plots

def attendance_vs_cocurricular_all(df):
    """
    Generates multiple visualizations for 
    Average_Attendance_On_Class vs Are_You_Engaged_With_Any_Co-Curriculum_Activities? (Yes/No).
    Returns Flask-friendly list of dicts {title, plot, note}.
    Includes static (matplotlib) + interactive (Plotly) charts.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    from io import BytesIO
    import base64

    plots = []

    # ✅ Check required columns
    if (
        "Average_Attendance_On_Class" not in df.columns or
        "Are_You_Engaged_With_Any_Co-Curriculum_Activities?" not in df.columns
    ):
        return [{"title": "Error", "plot": None, "note": "Missing required columns."}]

    # ✅ Data cleaning
    df = df.dropna(subset=[
        "Average_Attendance_On_Class", 
        "Are_You_Engaged_With_Any_Co-Curriculum_Activities?"
    ])
    df["Average_Attendance_On_Class"] = pd.to_numeric(
        df["Average_Attendance_On_Class"].astype(str).str.replace("%", "", regex=False),
        errors="coerce"
    )
    df = df.dropna(subset=["Average_Attendance_On_Class"])

    # ✅ Ensure consistent categorical order
    order = ["No", "Yes"] if df["Are_You_Engaged_With_Any_Co-Curriculum_Activities?"].nunique() == 2 else sorted(df["Are_You_Engaged_With_Any_Co-Curriculum_Activities?"].unique())


    # 1️⃣ Bar Plot (Mean Attendance)
    mean_df = (
        df.groupby("Are_You_Engaged_With_Any_Co-Curriculum_Activities?", as_index=False)["Average_Attendance_On_Class"]
        .mean()
        .sort_values(by="Average_Attendance_On_Class", ascending=False)
    )

    fig = px.bar(
        mean_df,
        x="Are_You_Engaged_With_Any_Co-Curriculum_Activities?",
        y="Average_Attendance_On_Class",
        color="Are_You_Engaged_With_Any_Co-Curriculum_Activities?",
        text_auto=".1f",
        category_orders={"Are_You_Engaged_With_Any_Co-Curriculum_Activities?": order},
        title="Interactive Bar — Mean Attendance by Co-curricular",
        labels={
            "Are_You_Engaged_With_Any_Co-Curriculum_Activities?": "Co-curricular Participation",
            "Average_Attendance_On_Class": "Average Attendance (%)"
        }
    )
    plots.append({
        "title": "Interactive Bar (Plotly)",
        "plot": fig.to_html(full_html=False),
        "note": "Hover for mean attendance per participation group."
    })

    # 3️⃣ Interactive Box Plot — Attendance by Co-curricular
    fig = px.box(
        df,
        x="Are_You_Engaged_With_Any_Co-Curriculum_Activities?",
        y="Average_Attendance_On_Class",
        color="Are_You_Engaged_With_Any_Co-Curriculum_Activities?",
        category_orders={"Are_You_Engaged_With_Any_Co-Curriculum_Activities?": order},
        points="outliers",
        title="Attendance vs Co-curricular - Box Plot",
        labels={
            "Are_You_Engaged_With_Any_Co-Curriculum_Activities?": "Co-curricular Participation",
            "Average_Attendance_On_Class": "Average Attendance (%)"
        }
    )
    plots.append({
        "title": "Attendance vs Co-curricular - Box Plot",
        "plot": fig.to_html(full_html=False),
        "note": "Shows spread, median, and outliers interactively across Yes/No groups."
    })


    # 5️⃣ (Optional) Interactive Bar Plot — Mean Attendance
   

    return plots

def attendance_vs_cgpa_all(df):
    """
    Generates multiple visualizations for Average_Attendance_On_Class vs What_Is_Your_Current_CGPA? (continuous).
    Returns a list of dictionaries with 'title', 'plot', and 'note'.
    Flask-friendly (base64 for static + HTML for interactive).
    """
    plots = []

    if "Average_Attendance_On_Class" not in df.columns or "What_Is_Your_Current_CGPA?" not in df.columns:
        return [{"title": "Error", "plot": None, "note": "Missing required columns."}]
    
    # 🧹 Data Cleaning
    df = df.dropna(subset=["Average_Attendance_On_Class", "What_Is_Your_Current_CGPA?"])
    df["Average_Attendance_On_Class"] = pd.to_numeric(
        df["Average_Attendance_On_Class"].astype(str).str.replace("%", "", regex=False),
        errors="coerce"
    )
    df["What_Is_Your_Current_CGPA?"] = pd.to_numeric(df["What_Is_Your_Current_CGPA?"], errors="coerce")
    df = df.dropna()

    # ------------------ 4️⃣ Interactive Scatter ------------------
    fig1 = px.scatter(
        df,
        x="Average_Attendance_On_Class",
        y="What_Is_Your_Current_CGPA?",
        trendline="ols",
        title="CGPA vs Attendance - Scatter plot with regression line",
        labels={"Average_Attendance_On_Class": "Average Attendance (%)", "What_Is_Your_Current_CGPA?": "CGPA"},
        color_discrete_sequence=["#1f77b4"]
    )
    plots.append({
        "title": "Interactive Scatter (Plotly)",
        "plot": fig1.to_html(full_html=False, include_plotlyjs="cdn"),
        "note": "Interactive version with zoom and hover info."
    })

    # ------------------ 5️⃣ Interactive Heatmap ------------------
   #import plotly.express as px

# pick only the two columns you want
    cols = ["Average_Attendance_On_Class", "What_Is_Your_Current_CGPA?"]

    # Create a short-name map for the heatmap only
    rename_map = {
        "Average_Attendance_On_Class": "Attendance",
        "What_Is_Your_Current_CGPA?": "CGPA"
    }

    # Copy only those columns and rename for display
    corr_df = df[cols].rename(columns=rename_map)

    # Compute correlation
    corr_matrix = corr_df.corr()

    # Plot the heatmap
    fig2 = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation Heatmap – Attendance vs CGPA"
    )

    fig2.update_layout(
        title_x=0.5,
        margin=dict(l=40, r=40, t=60, b=40),
        coloraxis_colorbar=dict(title="r-value")
    )

    plots.append({
        "title": "Attendance vs CGPA - Correlation Heatmap",
        "plot": fig2.to_html(full_html=False, include_plotlyjs="cdn"),
        "note": "Shows correlation strength between Attendance and CGPA. Names shortened only for readability."
    })

    return plots

def attendance_vs_feedback_all(df):
    """
    Generates multiple visualizations for Average_Attendance_On_Class vs Feedback_Rating (numeric).
    Returns a list of dictionaries with 'title', 'plot', and 'note'.
    Flask-friendly (base64 for static + HTML for interactive).
    """
    plots = []

    if "Average_Attendance_On_Class" not in df.columns or "Feedback_Rating" not in df.columns:
        return [{"title": "Error", "plot": None, "note": "Missing required columns."}]

    # 🧹 Data Cleaning
    df = df.dropna(subset=["Average_Attendance_On_Class", "Feedback_Rating"])
    df["Average_Attendance_On_Class"] = pd.to_numeric(
        df["Average_Attendance_On_Class"].astype(str).str.replace("%", "", regex=False),
        errors="coerce"
    )
    df["Feedback_Rating"] = pd.to_numeric(df["Feedback_Rating"], errors="coerce")
    df = df.dropna()

    # ------------------ 4️⃣ Interactive Scatter ------------------
    fig1 = px.scatter(
        df,
        x="Average_Attendance_On_Class",
        y="Feedback_Rating",
        trendline="ols",
        title="Interactive Scatter with Regression Line",
        labels={"Average_Attendance_On_Class": "Average Attendance (%)", "Feedback_Rating": "Feedback Rating"},
        color_discrete_sequence=["#9b59b6"]
    )
    plots.append({
        "title": "Interactive Scatter (Plotly)",
        "plot": fig1.to_html(full_html=False, include_plotlyjs="cdn"),
        "note": "Interactive view with regression line and hover info."
    })

    # ------------------ 5️⃣ Interactive Heatmap ------------------

# Select only the numeric columns you mentioned
    cols = ["Average_Attendance_On_Class", "Feedback_Rating"]

    # Create a short-name map for the heatmap only
    rename_map = {
        "Average_Attendance_On_Class": "Attendance",
        "Feedback_Rating": "Feedback"
    }

    # Copy only those columns and rename for display
    corr_df = df[cols].rename(columns=rename_map)

    # Compute correlation
    corr_matrix = corr_df.corr()

    # Plot the heatmap
    fig2 = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation Heatmap – Attendance vs Feedback"
    )

    fig2.update_layout(
        title_x=0.5,
        margin=dict(l=40, r=40, t=60, b=40),
        coloraxis_colorbar=dict(title="r-value")
    )

    plots.append({
        "title": "Attendance vs Feedback Rating - Correlation Heatmap",
        "plot": fig2.to_html(full_html=False, include_plotlyjs="cdn"),
        "note": "Shows correlation strength between Attendance and Feedback. Names shortened only for readability."
    })


   

    return plots

def attendance_vs_teacher_feedback_all(df):
    """
    Generates multiple visualizations for Average_Attendance_On_Class vs Teacher_Feedback (categorical).
    Returns a list of dictionaries with 'title', 'plot', and 'note'.
    Flask-friendly (base64 for static + HTML for interactive).
    """
    plots = []

    # ✅ Check required columns
    required_cols = {"Average_Attendance_On_Class", "Teacher_Feedback"}
    if not required_cols.issubset(df.columns):
        return [{"title": "Error", "plot": None, "note": f"Missing required columns: {required_cols - set(df.columns)}"}]

    # 🧹 Clean Data
    df = df.dropna(subset=["Average_Attendance_On_Class", "Teacher_Feedback"]).copy()
    df["Average_Attendance_On_Class"] = pd.to_numeric(
        df["Average_Attendance_On_Class"].astype(str).str.replace("%", "", regex=False),
        errors="coerce"
    )
    df = df.dropna(subset=["Average_Attendance_On_Class"])

    if df.empty:
        return [{"title": "No valid data", "plot": None, "note": "No rows available after cleaning."}]

    # ✅ Helper function to convert matplotlib figure to base64

    # ------------------ 3️⃣ Mean Attendance Bar Plot ------------------

# ✅ Compute mean attendance per feedback
    mean_df = df.groupby("Teacher_Feedback", as_index=False)["Average_Attendance_On_Class"].mean()

    # ✅ Create interactive bar plot
    fig = px.bar(
        mean_df,
        x="Teacher_Feedback",
        y="Average_Attendance_On_Class",
        color="Teacher_Feedback",
        color_continuous_scale="Viridis",
        title="Average Attendance vs Teacher Feedback – Mean Attendance",
        text_auto=".2f"
    )

    # ✅ Customize layout for better readability
    fig.update_layout(
        xaxis_title="Teacher Feedback",
        yaxis_title="Mean Attendance (%)",
        xaxis_tickangle=30,
        coloraxis_showscale=False,
        template="plotly_white",
        height=450
    )

    # ✅ Append to your plots list
    plots.append({
        "title": "Interactive Bar (Plotly)",
        "plot": fig.to_html(full_html=False),
        "note": "Displays the average attendance per feedback group interactively."
    })


    # ------------------ 4️⃣ Interactive Box Plot (Plotly) ------------------
    fig_box = px.box(
        df,
        x="Teacher_Feedback",
        y="Average_Attendance_On_Class",
        color="Teacher_Feedback",
        title="Interactive Box Plot – Attendance by Feedback",
        labels={"Average_Attendance_On_Class": "Average Attendance (%)"},
        color_discrete_sequence=px.colors.qualitative.Set2,
        points="all"
    )
    plots.append({
        "title": "Interactive Box Plot (Plotly)",
        "plot": fig_box.to_html(full_html=False, include_plotlyjs="cdn"),
        "note": "Interactive visualization for exploring attendance spread across feedback levels."
    })


    return plots

def cgpa_vs_cocurricular_all(df):
    """
    Generates multiple visualizations for What_Is_Your_Current_CGPA? vs Are_You_Engaged_With_Any_Co-Curriculum_Activities?.
    Returns a list of dicts with 'title', 'plot', and 'note'.
    Flask-friendly (base64 for static + HTML for interactive).
    """

    import pandas as pd
    import plotly.express as px
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO

    plots = []

    # ✅ Required columns
    required_cols = {"What_Is_Your_Current_CGPA?", "Are_You_Engaged_With_Any_Co-Curriculum_Activities?"}
    if not required_cols.issubset(df.columns):
        return [{"title": "Error", "plot": None,
                 "note": f"Missing required columns: {required_cols - set(df.columns)}"}]

    # 🧹 Clean data
    df = df.dropna(subset=list(required_cols)).copy()
    df["What_Is_Your_Current_CGPA?"] = pd.to_numeric(df["What_Is_Your_Current_CGPA?"], errors="coerce")
    df = df.dropna(subset=["What_Is_Your_Current_CGPA?"])

    if df.empty:
        return [{"title": "No valid data", "plot": None, "note": "No rows available after cleaning."}]

    # Helper for matplotlib → base64
    def fig_to_base64(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    # --------------------------------------------------------------------
    # 1️⃣ Mean CGPA Bar Plot (Plotly)
    mean_df = df.groupby("Are_You_Engaged_With_Any_Co-Curriculum_Activities?", as_index=False)["What_Is_Your_Current_CGPA?"].mean()

    fig = px.bar(
        mean_df,
        x="Are_You_Engaged_With_Any_Co-Curriculum_Activities?",
        y="What_Is_Your_Current_CGPA?",
        color="Are_You_Engaged_With_Any_Co-Curriculum_Activities?",
        color_continuous_scale="Plasma",
        text_auto=".2f",
        title="CGPA vs Co-Curricular - Mean Bar"
    )
    fig.update_layout(
        xaxis_title="Co-Curricular Activities (Engaged?)",
        yaxis_title="Mean CGPA",
        xaxis_tickangle=30,
        coloraxis_showscale=False,
        template="plotly_white",
        height=450
    )

    plots.append({
        "title": "Interactive Mean CGPA Bar (Plotly)",
        "plot": fig.to_html(full_html=False),
        "note": "Shows average CGPA for each Co-Curricular activity category."
    })

    # --------------------------------------------------------------------
    # 5️⃣ KDE Density Plot (Seaborn)
    if df["Are_You_Engaged_With_Any_Co-Curriculum_Activities?"].nunique() == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        for group, data in df.groupby("Are_You_Engaged_With_Any_Co-Curriculum_Activities?"):
            sns.kdeplot(data["What_Is_Your_Current_CGPA?"], fill=True, label=str(group), ax=ax)
        ax.legend(title="Engaged?")
        ax.set_title("CGPA vs Co-Curricular - KDE")
        ax.set_xlabel("CGPA")
        ax.set_ylabel("Density")
        img_b64 = fig_to_base64(fig)
        plt.close(fig)
        plots.append({
            "title": "CGPA vs Co-Curricular - KDE",
            "plot": f"<img src='data:image/png;base64,{img_b64}' class='img-fluid'/>",
            "note": "Shows distribution overlap of CGPA across activity groups (if binary categories)."
        })

    return plots

def feedback_vs_cgpa_all(df):
    """
    Generates multiple visualizations for Feedback_Rating vs What_Is_Your_Current_CGPA?.
    Returns a list of dicts with 'title', 'plot', and 'note'.
    Flask-friendly (base64 for static + HTML for interactive).
    """

    import pandas as pd
    import plotly.express as px
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO
    from scipy.stats import pearsonr

    plots = []

    # ✅ Required columns
    required_cols = {"Feedback_Rating", "What_Is_Your_Current_CGPA?"}
    if not required_cols.issubset(df.columns):
        return [{
            "title": "Error",
            "plot": None,
            "note": f"Missing required columns: {required_cols - set(df.columns)}"
        }]

    # 🧹 Clean data
    df = df.dropna(subset=list(required_cols)).copy()
    df["Feedback_Rating"] = pd.to_numeric(df["Feedback_Rating"], errors="coerce")
    df["What_Is_Your_Current_CGPA?"] = pd.to_numeric(df["What_Is_Your_Current_CGPA?"], errors="coerce")
    df = df.dropna(subset=["Feedback_Rating", "What_Is_Your_Current_CGPA?"])

    if df.empty:
        return [{
            "title": "No valid data",
            "plot": None,
            "note": "No rows available after cleaning."
        }]

    # Helper for matplotlib → base64
    def fig_to_base64(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    # --------------------------------------------------------------------
    # 1️⃣ Scatter Plot (Plotly)
    fig = px.scatter(
        df,
        x="Feedback_Rating",
        y="What_Is_Your_Current_CGPA?",
        color="Feedback_Rating",
        color_continuous_scale="Viridis",
        title="Feedback Rating vs CGPA – Scatter Plot",
        trendline="ols",
        labels={
            "Feedback_Rating": "Feedback Rating",
            "What_Is_Your_Current_CGPA?": "CGPA"
        }
    )
    fig.update_layout(template="plotly_white", height=450)

    plots.append({
        "title": "Interactive Scatter Plot (Plotly)",
        "plot": fig.to_html(full_html=False, include_plotlyjs="cdn"),
        "note": "Scatter plot showing relationship between feedback ratings and CGPA with regression trendline."
    })

    # --------------------------------------------------------------------
    # 2️⃣ Correlation Heatmap (Seaborn)
    cols = ["What_Is_Your_Current_CGPA?", "Feedback_Rating"]

    # Create a short-name map for the heatmap only
    rename_map = {
        "What_Is_Your_Current_CGPA?": "Attendance",
        "Feedback_Rating": "Feedback"
    }

    # Copy only those columns and rename for display
    corr_df = df[cols].rename(columns=rename_map)

    # Compute correlation
    corr_matrix = corr_df.corr()

    # Plot the heatmap
    fig2 = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation Heatmap – Attendance vs CGPA"
    )

    fig2.update_layout(
        title_x=0.5,
        margin=dict(l=40, r=40, t=60, b=40),
        coloraxis_colorbar=dict(title="r-value")
    )

    plots.append({
        "title": "Attendance vs CGPA - Correlation Heatmap",
        "plot": fig2.to_html(full_html=False, include_plotlyjs="cdn"),
        "note": "Shows correlation strength between Attendance and CGPA. Names shortened only for readability."
    })



    # --------------------------------------------------------------------
    # # 3️⃣ Regression Plot (Seaborn)
    # fig, ax = plt.subplots(figsize=(6, 4))
    # sns.regplot(
    #     data=df,
    #     x="Feedback_Rating",
    #     y="What_Is_Your_Current_CGPA?",
    #     scatter_kws={'alpha': 0.6},
    #     line_kws={"color": "red"},
    #     ax=ax
    # )
    # ax.set_title("Regression Line: Feedback Rating vs CGPA")
    # ax.set_xlabel("Feedback Rating")
    # ax.set_ylabel("CGPA")
    # img_b64 = fig_to_base64(fig)
    # plt.close(fig)
    # plots.append({
    #     "title": "Static Regression Plot (Seaborn)",
    #     "plot": f"<img src='data:image/png;base64,{img_b64}' class='img-fluid'/>",
    #     "note": "Regression plot visualizing linear trend between Feedback Rating and CGPA."
    # })

    # --------------------------------------------------------------------
    
    return plots

def teacher_feedback_vs_cgpa_all(df):
    """
    Generates multiple visualizations for Teacher_Feedback (categorical) vs What_Is_Your_Current_CGPA? (numeric).
    Returns a list of dicts with 'title', 'plot', and 'note'.
    Flask-friendly (base64 for static + HTML for interactive).
    """

    import pandas as pd
    import plotly.express as px
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO

    plots = []

    # ✅ Required columns
    required_cols = {"Teacher_Feedback", "What_Is_Your_Current_CGPA?"}
    if not required_cols.issubset(df.columns):
        return [{
            "title": "Error",
            "plot": None,
            "note": f"Missing required columns: {required_cols - set(df.columns)}"
        }]

    # 🧹 Clean data
    df = df.dropna(subset=list(required_cols)).copy()
    df["What_Is_Your_Current_CGPA?"] = pd.to_numeric(df["What_Is_Your_Current_CGPA?"], errors="coerce")
    df = df.dropna(subset=["What_Is_Your_Current_CGPA?"])

    if df.empty:
        return [{
            "title": "No valid data",
            "plot": None,
            "note": "No rows available after cleaning."
        }]

    # Helper to convert matplotlib figs → base64
    def fig_to_base64(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    # --------------------------------------------------------------------
    # 1️⃣ Mean CGPA Bar Plot (Plotly)
    mean_df = df.groupby("Teacher_Feedback", as_index=False)["What_Is_Your_Current_CGPA?"].mean()
    agg_df = df.groupby("Teacher_Feedback", as_index=False)["What_Is_Your_Current_CGPA?"].agg(['mean', 'std']).reset_index()
    fig = px.bar(
        agg_df,
        x="Teacher_Feedback",
        y="mean",
        error_y="std",
        color="Teacher_Feedback",
        title="CGPA vs Teacher Feedback - Std Mean bar",
        labels={"mean": "Mean CGPA", "Teacher_Feedback": "Teacher Feedback"},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_layout(template="plotly_white", height=450)
    plots.append({
        "title": "CGPA vs Teacher Feedback - Std Mean bar",
        "plot": fig.to_html(full_html=False, include_plotlyjs="cdn"),
        "note": "Shows mean CGPA with standard deviation error bars for each feedback category."
    })

    # fig = px.bar(
    #     mean_df,
    #     x="Teacher_Feedback",
    #     y="What_Is_Your_Current_CGPA?",
    #     color="Teacher_Feedback",
    #     color_continuous_scale="Viridis",
    #     text_auto=".2f",
    #     title="Mean CGPA vs Teacher Feedback"
    # )
    # fig.update_layout(
    #     xaxis_title="Teacher Feedback",
    #     yaxis_title="Mean CGPA",
    #     xaxis_tickangle=30,
    #     coloraxis_showscale=False,
    #     template="plotly_white",
    #     height=450
    # )

    # plots.append({
    #     "title": "Interactive Mean CGPA Bar (Plotly)",
    #     "plot": fig.to_html(full_html=False, include_plotlyjs="cdn"),
    #     "note": "Shows average CGPA per Teacher Feedback category interactively."
    # })

    # --------------------------------------------------------------------
    # 2️⃣ Interactive Box Plot (Plotly)
    fig_box = px.box(
        df,
        x="Teacher_Feedback",
        y="What_Is_Your_Current_CGPA?",
        color="Teacher_Feedback",
        title="CGPA Distribution by Teacher Feedback (Box Plot)",
        labels={"What_Is_Your_Current_CGPA?": "CGPA"},
        color_discrete_sequence=px.colors.qualitative.Set2,
        points="all"
    )
    fig_box.update_layout(template="plotly_white", height=450)
    plots.append({
        "title": "Interactive Box Plot (Plotly)",
        "plot": fig_box.to_html(full_html=False, include_plotlyjs="cdn"),
        "note": "Interactive visualization for CGPA distribution across feedback levels."
    })

    # --------------------------------------------------------------------
    # 5️⃣ Bar + Error Bars (Mean ± Std)
   

    return plots

def cocurricular_vs_teacher_feedback_all(df):
    """
    Generates multiple visualizations for
    Are_You_Engaged_With_Any_Co-Curriculum_Activities? (categorical)
    vs Teacher_Feedback (categorical).
    
    Returns a list of dicts with 'title', 'plot', and 'note'.
    Flask-friendly (base64 for static + HTML for interactive).
    """

    import pandas as pd
    import plotly.express as px
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO

    plots = []

    # ✅ Required columns
    required_cols = {
        "Are_You_Engaged_With_Any_Co-Curriculum_Activities?",
        "Teacher_Feedback"
    }
    if not required_cols.issubset(df.columns):
        return [{
            "title": "Error",
            "plot": None,
            "note": f"Missing required columns: {required_cols - set(df.columns)}"
        }]

    # 🧹 Clean data
    df = df.dropna(subset=list(required_cols)).copy()
    df = df[df["Are_You_Engaged_With_Any_Co-Curriculum_Activities?"].astype(str).str.strip() != ""]
    df = df[df["Teacher_Feedback"].astype(str).str.strip() != ""]

    if df.empty:
        return [{
            "title": "No valid data",
            "plot": None,
            "note": "No rows available after cleaning."
        }]

    # --------------------------------------------------------------------
    # 1️⃣ Count Plot (Seaborn)

    # --------------------------------------------------------------------
    # 2️⃣ Interactive Grouped Bar Chart (Plotly)
    count_df = df.groupby(
        ["Are_You_Engaged_With_Any_Co-Curriculum_Activities?", "Teacher_Feedback"]
    ).size().reset_index(name="Count")


    # --------------------------------------------------------------------
    # 3️⃣ Crosstab Heatmap (Seaborn)
    cross = pd.crosstab(
        df["Are_You_Engaged_With_Any_Co-Curriculum_Activities?"],
        df["Teacher_Feedback"],
        normalize="index"
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cross, annot=True, cmap="YlGnBu", fmt=".2f", cbar_kws={'label': 'Proportion'}, ax=ax)
    ax.set_title("Proportion Heatmap: Co-Curricular Activities × Teacher Feedback")
    ax.set_xlabel("Teacher Feedback")
    ax.set_ylabel("Co-Curricular Activities")
    img_b64 = fig_to_base64(fig)
    plt.close(fig)

    plots.append({
        "title": "Static Proportion Heatmap (Seaborn)",
        "plot": f"<img src='data:image/png;base64,{img_b64}' class='img-fluid'/>",
        "note": "Heatmap showing proportions of feedback categories within each activity group."
    })

    # --------------------------------------------------------------------
    # 4️⃣ Interactive Stacked Bar (Plotly)
    cross_abs = pd.crosstab(
        df["Are_You_Engaged_With_Any_Co-Curriculum_Activities?"],
        df["Teacher_Feedback"]
    )

    stacked_df = cross_abs.div(cross_abs.sum(axis=1), axis=0).reset_index()
    stacked_melt = stacked_df.melt(
        id_vars="Are_You_Engaged_With_Any_Co-Curriculum_Activities?",
        var_name="Teacher_Feedback",
        value_name="Proportion"
    )

    fig = px.bar(
        stacked_melt,
        x="Are_You_Engaged_With_Any_Co-Curriculum_Activities?",
        y="Proportion",
        color="Teacher_Feedback",
        title="Stacked Bar: Teacher Feedback Distribution by Co-Curricular Activities",
        text_auto=".1%",
        barmode="stack"
    )
    fig.update_layout(
        template="plotly_white",
        height=450,
        yaxis=dict(title="Proportion", tickformat=".0%")
    )

    plots.append({
        "title": "Interactive Stacked Bar (Plotly)",
        "plot": fig.to_html(full_html=False, include_plotlyjs="cdn"),
        "note": "Shows relative distribution (proportions) of feedback levels within each activity group."
    })

    # --------------------------------------------------------------------
    # 5️⃣ Chi-Square Statistical Association
    try:
        from scipy.stats import chi2_contingency
        chi2, p, dof, expected = chi2_contingency(pd.crosstab(
            df["Are_You_Engaged_With_Any_Co-Curriculum_Activities?"],
            df["Teacher_Feedback"]
        ))
        note = f"Chi-square test: χ² = {chi2:.2f}, p-value = {p:.4f} → {'Significant' if p < 0.05 else 'Not significant'} association."
    except Exception:
        note = "Unable to compute Chi-square association test."

    plots.append({
        "title": "Statistical Association (Chi-Square Test)",
        "plot": None,
        "note": note
    })

    return plots

def gender_vs_feedback_rating_all(df):
    """
    Gender vs Feedback_Rating visualizations.
    Flask-friendly: returns list of dicts with 'title', 'plot', 'note'.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    from io import BytesIO
    import base64
    import pandas as pd

    plots = []

    if 'Gender' not in df.columns or 'Feedback_Rating' not in df.columns:
        return [{"title": "Error", "plot": None, "note": "Missing 'Gender' or 'Feedback_Rating' columns"}]

    # Clean + numeric feedback
    df = df.dropna(subset=['Gender', 'Feedback_Rating'])
    df = df[df['Gender'].astype(str).str.strip() != ""]
    df['Feedback_Rating'] = pd.to_numeric(df['Feedback_Rating'], errors='coerce')
    df = df.dropna(subset=['Feedback_Rating'])

    # 1️⃣ Mean rating by gender (Plotly)
    avg_feedback = (
        df.groupby('Gender')['Feedback_Rating']
        .agg(['mean', 'std'])
        .reset_index()
        .rename(columns={'mean': 'Average_Rating', 'std': 'Std_Dev'})
    )

    fig1 = px.bar(
        avg_feedback,
        x='Gender',
        y='Average_Rating',
        color='Gender',
        error_y='Std_Dev',
        title="Gender vs Average Feedback Rating",
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig1.update_layout(title_x=0.5, height=450)
    plots.append({
        "title": "Average Feedback Rating by Gender",
        "plot": fig1.to_html(full_html=False),
        "note": "Mean feedback ratings with standard deviation error bars."
    })

    # 2️⃣ Distribution KDE
    # 2️⃣ Distribution KDE
    plt.figure(figsize=(6, 4))

    # Ensure Feedback_Rating is numeric
    df['Feedback_Rating'] = pd.to_numeric(df['Feedback_Rating'], errors='coerce')
    df = df.dropna(subset=['Feedback_Rating', 'Gender'])

    for g in df['Gender'].unique():
        sns.kdeplot(df[df['Gender'] == g]['Feedback_Rating'], fill=True, label=g)

    plt.title("Feedback Rating Distribution by Gender")
    plt.xlabel("Feedback Rating")
    plt.ylabel("Density")
    plt.legend()

    # Convert plot to base64 image and embed in full <img> HTML tag
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)

    # Base64 string
    encoded_img = base64.b64encode(buf.getvalue()).decode()

    # Append HTML-ready image instead of raw string
    plots.append({
        "title": "KDE Distribution",
        "plot": f"<img src='data:image/png;base64,{encoded_img}' alt='KDE Distribution' class='img-fluid rounded shadow'>",
        "note": "Smooth comparison of feedback rating distribution by gender."
    })

    plt.close()

    # 3️⃣ Interactive Box Plot
    fig2 = px.box(df, x='Gender', y='Feedback_Rating', color='Gender',
                  title="Box Plot – Gender vs Feedback Rating")
    plots.append({
        "title": "Interactive Box Plot",
        "plot": fig2.to_html(full_html=False),
        "note": "Shows spread and outliers of ratings per gender."
    })

    return plots

def gender_vs_teacher_feedback_all(df):
    """
    Gender vs Teacher_Feedback (categorical) visualizations.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    from io import BytesIO
    import base64
    import pandas as pd

    plots = []

    if 'Gender' not in df.columns or 'Teacher_Feedback' not in df.columns:
        return [{"title": "Error", "plot": None, "note": "Missing 'Gender' or 'Teacher_Feedback' columns"}]

    df = df.dropna(subset=['Gender', 'Teacher_Feedback'])
    df = df[(df['Gender'].astype(str).str.strip() != "") & (df['Teacher_Feedback'].astype(str).str.strip() != "")]

    counts = df.groupby(['Teacher_Feedback', 'Gender']).size().reset_index(name='Count')

    # 1️⃣ Stacked Bar
    fig1 = px.bar(
        counts, x='Teacher_Feedback', y='Count', color='Gender',
        barmode='stack', title="Gender Count vs Teacher Feedback"
    )
    fig1.update_layout(title_x=0.5, height=450)
    plots.append({
        "title": "Interactive Stacked Bar",
        "plot": fig1.to_html(full_html=False),
        "note": "Stacked bar showing gender-wise counts for each feedback type."
    })

    # 2️⃣ Grouped Bar
    fig2 = px.bar(counts, x='Teacher_Feedback', y='Count', color='Gender',
                  barmode='group', title="Grouped Bar – Gender vs Teacher Feedback")
    plots.append({
        "title": "Interactive Grouped Bar",
        "plot": fig2.to_html(full_html=False),
        "note": "Grouped bars showing comparative gender counts per feedback."
    })

    # 3️⃣ Heatmap
    pivot = counts.pivot(index='Teacher_Feedback', columns='Gender', values='Count').fillna(0)
    fig3 = px.imshow(pivot, text_auto=True, color_continuous_scale="YlGnBu",
                     title="Heatmap – Gender Count per Teacher Feedback")
    plots.append({
        "title": "Interactive Heatmap",
        "plot": fig3.to_html(full_html=False, include_plotlyjs='cdn'),
        "note": "Interactive heatmap of gender counts across feedback categories."
    })

    # 4️⃣ Sunburst
    fig4 = px.sunburst(df, path=['Teacher_Feedback', 'Gender'],
                       title="Sunburst: Teacher Feedback → Gender")
    plots.append({
        "title": "Sunburst Chart",
        "plot": fig4.to_html(full_html=False),
        "note": "Hierarchical relationship from feedback type to gender."
    })

    return plots

def dept_vs_co_curricular_all(df):
    """
    Department vs Co-Curricular Activities visualizations.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    from io import BytesIO
    import base64
    import pandas as pd

    plots = []
    col_dept = 'Department'
    col_coc = 'Are_You_Engaged_With_Any_Co-Curriculum_Activities?'

    if col_dept not in df.columns or col_coc not in df.columns:
        return [{"title": "Error", "plot": None, "note": "Missing required columns"}]

    df = df.dropna(subset=[col_dept, col_coc])
    df[col_coc] = df[col_coc].astype(str).str.strip().str.title()

    counts = df.groupby([col_dept, col_coc]).size().unstack(fill_value=0)

   

    # 2️⃣ Stacked Bar
    df_long = counts.reset_index().melt(id_vars=col_dept, var_name=col_coc, value_name='Count')
    fig2 = px.bar(df_long, x=col_dept, y='Count', color=col_coc,
                  barmode='stack', title="Department vs Co-Curricular (Stacked)")
    plots.append({
        "title": "Interactive Stacked Bar",
        "plot": fig2.to_html(full_html=False),
        "note": "Stacked bar comparing participation levels per department."
    })


    # 3️⃣ Sunburst
    fig3 = px.sunburst(df, path=[col_dept, col_coc],
                       title="Sunburst – Department → Co-Curricular")
    plots.append({
        "title": "Sunburst Chart",
        "plot": fig3.to_html(full_html=False),
        "note": "Hierarchical breakdown of departments and participation."
    })

    return plots

def cocurricular_vs_teacher_feedback_all(df):

    """
    Co-Curricular Activities vs Teacher_Feedback visualizations.
    Flask-friendly and fully self-contained.
    """
    import pandas as pd
    import plotly.express as px
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO

    def fig_to_base64(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode()

    plots = []
    col_coc = "Are_You_Engaged_With_Any_Co-Curriculum_Activities?"
    col_fb = "Teacher_Feedback"

    if col_coc not in df.columns or col_fb not in df.columns:
        return [{"title": "Error", "plot": None, "note": "Missing required columns"}]

    df = df.dropna(subset=[col_coc, col_fb])
    df[col_coc] = df[col_coc].astype(str).str.strip().str.title()
    df[col_fb] = df[col_fb].astype(str).str.strip().str.title()

    # 2️⃣ Interactive Stacked Bar
    cross_abs = pd.crosstab(df[col_coc], df[col_fb])
    stacked = cross_abs.div(cross_abs.sum(axis=1), axis=0).reset_index()
    stacked_melt = stacked.melt(id_vars=col_coc, var_name=col_fb, value_name='Proportion')

    fig2 = px.bar(stacked_melt, x=col_coc, y='Proportion', color=col_fb,
                  title="Teacher Feedback Distribution by Co-Curricular",
                  barmode='stack', text_auto=".1%")
    fig2.update_layout(template="plotly_white", height=450, yaxis=dict(tickformat=".0%"))
    plots.append({
        "title": "Interactive Stacked Bar",
        "plot": fig2.to_html(full_html=False),
        "note": "Shows relative proportions of feedback by activity involvement."
    })

    return plots

def correlation_and_scatter_all(df):
    """
    Generates correlation-based and scatter plots for all numeric columns.
    Returns a list of dicts with 'title', 'plot', and 'note'.
    Flask-friendly: static plots as base64 images, interactive as HTML.
    """

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    from io import BytesIO
    import base64
    import itertools

    plots = []
    num_cols = df.select_dtypes(include='number').columns.tolist()
    # ✅ Select only numeric columns:
    for col_x, col_y in itertools.combinations(num_cols, 2):
        fig_scatter = px.scatter(
            df,
            x=col_x,
            y=col_y,
            trendline="ols",
            title=f"{col_x} vs {col_y}",
            opacity=0.7
        )
        plots.append({
            "title": f"Scatter Plot: {col_x} vs {col_y}",
            "plot": fig_scatter.to_html(full_html=False),
            "note": f"Examines how {col_x} and {col_y} relate linearly and visually"
        })


    # --- INTERACTIVE CORRELATION HEATMAP (Plotly) ---
    corr = df[num_cols].corr().round(2)
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Interactive Correlation Heatmap")
    plots.append({
        "title": "Interactive Correlation Heatmap",
        "plot": fig_corr.to_html(full_html=False),
        "note": "Interactive version of correlation heatmap"
    })

    # --- SCATTER PLOTS (Plotly) ---
   

    return plots

