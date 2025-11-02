from flask import Flask, render_template, request, redirect, jsonify, url_for, session, flash
import sys
import os
import pandas as pd
import json
import time
from functools import lru_cache
import hashlib

# Add the parent directory to Python path to import analysis module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from analysis import *
except ImportError as e:
    print(f"Error importing analysis module: {e}")
    # minimal dummy fallbacks (keeps app running)
    import pandas as _pd
    def load_data(dataset_id="default"):
        return _pd.DataFrame(), {"name": "Test Dataset", "rows": 0, "columns": 0}
    def univariate(df): return {}
    def sunburst_chart(df): return "<p>Sunburst unavailable</p>"
    def gender_vs_department_all(df): return []
    def gender_vs_avg_attendance_all(df): return []
    def gender_vs_cocurricular_all(df): return []
    def gender_vs_feedback_rating_all(df): return []
    def gender_vs_teacher_feedback_all(df): return []
    def gender_vs_cgpa(df): return []
    def dept_vs_avg_attendance(df): return []
    def dept_vs_co_curricular_all(df): return []
    def dept_vs_cgpa_all(df): return []
    def dept_vs_feedback_all(df): return []
    def dept_vs_teacher_feedback(df): return []
    def attendance_vs_cocurricular_all(df): return []
    def attendance_vs_cgpa_all(df): return []
    def attendance_vs_feedback_all(df): return []
    def attendance_vs_teacher_feedback_all(df): return []
    def cgpa_vs_cocurricular_all(df): return []
    def feedback_vs_cgpa_all(df): return []
    def teacher_feedback_vs_cgpa_all(df): return []
    def add_uploaded_dataset(file, filename):
        raise RuntimeError("Upload not available in dummy mode")
    def get_datasets():
        return {"default": {"name": "Default Dataset", "filename": "data.csv", "path": "default/data.csv", "rows":0, "columns":0, "is_default": True}}
    def generate_insights(): return {"title": "Insights", "content": "<p>No insights available</p>"}
    def generate_insights_Academic(): return {"title": "Academic Insights", "content": "<p>No academic insights available</p>"}
    def generate_insights_Attendance(): return {"title": "Attendance Insights", "content": "<p>No attendance insights available</p>"}
    def generate_insights_Feedback(): return {"title": "Feedback Insights", "content": "<p>No feedback insights available</p>"}
    def generate_insights_Engagement(): return {"title": "Engagement Insights", "content": "<p>No engagement insights available</p>"}
    def generate_insights_Correlation(): return {"title": "Correlation Insights", "content": "<p>No correlation insights available</p>"}

app = Flask(__name__, template_folder="../templates")
app.secret_key = 'S_h$kY7!s#cR*t_K3y_For_Gemini_App'

# Admin credential examples
ADMIN_CREDENTIALS = {
    'admin1': 'pass123',
    'admin_dev': 'dev_pass',
    'manager': 'secure_mngr',
    'supervisor': 'supervise1',
    'root': 'toor4life'
}

# -------------------------
# ADVANCED CACHE SYSTEM for Performance
# -------------------------
class AdvancedDataCache:
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
        self._hits = 0
        self._misses = 0
    
    def get(self, key, max_age=300):  # 5 minutes default
        if key in self._cache:
            if time.time() - self._timestamps[key] < max_age:
                self._hits += 1
                return self._cache[key]
            else:
                # Expired
                del self._cache[key]
                del self._timestamps[key]
                self._misses += 1
        else:
            self._misses += 1
        return None
    
    def set(self, key, value):
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def clear_dataset_cache(self, dataset_id):
        """Clear all cache entries for a specific dataset"""
        keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"dataset_{dataset_id}")]
        for key in keys_to_remove:
            del self._cache[key]
            del self._timestamps[key]
        print(f"🧹 Cleared cache for dataset: {dataset_id}")
    
    def clear_all(self):
        """Clear entire cache"""
        self._cache.clear()
        self._timestamps.clear()
        print("🧹 Cleared all cache")
    
    def get_stats(self):
        """Get cache statistics"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'size': len(self._cache)
        }

# Global cache instance
data_cache = AdvancedDataCache()

# -------------------------
# Optimized Helper utilities
# -------------------------
def safe_execute(func, df, default=None, cache_key=None, use_cache=True):
    """Safely execute analysis function with optional caching and performance tracking."""
    if use_cache and cache_key:
        cached_result = data_cache.get(cache_key)
        if cached_result is not None:
            print(f"✅ {func.__name__} - CACHE HIT")
            return cached_result
    
    try:
        start_time = time.time()
        result = func(df)
        execution_time = time.time() - start_time
        
        if execution_time > 1.0:
            print(f"⚠️ {func.__name__} executed in {execution_time:.2f}s (SLOW)")
        else:
            print(f"✅ {func.__name__} executed in {execution_time:.2f}s")
        
        if use_cache and cache_key and result is not None:
            data_cache.set(cache_key, result)
            
        return result
    except Exception as e:
        print(f"❌ Error in {func.__name__}: {e}")
        return default if default is not None else []

def generate_cache_key(func_name, df, filters=None, dataset_id=None):
    """Generate a unique cache key based on function, data, and filters"""
    key_data = {
        'func': func_name,
        'dataset': dataset_id,
        'data_shape': df.shape,
        'data_hash': hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:16],
        'filters': filters
    }
    return f"{func_name}_{dataset_id}_{hash(str(key_data))}"

def standardize_plot_format(plots_data, default_title="Plot"):
    """Convert dict output from univariate into a display-friendly dict (if needed)."""
    if isinstance(plots_data, dict):
        return plots_data
    out = {}
    if isinstance(plots_data, list):
        for idx, item in enumerate(plots_data):
            title = item.get("title", f"{default_title} {idx+1}")
            out[title] = item.get("plot", "")
    return out

def apply_filters(df, filters):
    """Apply filters to dataframe based on filter criteria - OPTIMIZED"""
    if not filters or all(v == 'all' for v in filters.values()):
        return df
    
    filtered_df = df.copy()
    
    try:
        # Gender filter
        if filters.get('gender') and filters['gender'] != 'all':
            filtered_df = filtered_df[filtered_df['Gender'].astype(str).str.strip() == filters['gender'].strip()]
        
        # Department filter
        if filters.get('department') and filters['department'] != 'all':
            filtered_df = filtered_df[filtered_df['Department'].astype(str).str.strip() == filters['department'].strip()]
        
        # Co-curricular filter
        if filters.get('cocurricular') and filters['cocurricular'] != 'all':
            filtered_df = filtered_df[filtered_df['Are_You_Engaged_With_Any_Co-Curriculum_Activities?'].astype(str).str.strip() == filters['cocurricular'].strip()]
        
        # Teacher Feedback filter
        if filters.get('teacher_feedback') and filters['teacher_feedback'] != 'all':
            filtered_df = filtered_df[filtered_df['Teacher_Feedback'].astype(str).str.strip() == filters['teacher_feedback'].strip()]
            
    except Exception as e:
        print(f"Error applying filters: {e}")
    
    return filtered_df

def get_filter_options(df):
    """Get available options for filters from the dataset - OPTIMIZED with caching"""
    cache_key = f"filter_options_{hash(str(df.columns) + str(len(df)))}"
    cached = data_cache.get(cache_key)
    if cached:
        return cached
        
    options = {
        'genders': ['all'],
        'departments': ['all'],
        'cocurriculars': ['all'],
        'teacher_feedbacks': ['all']
    }
    
    try:
        # Gender options
        gender_cols = [col for col in df.columns if 'gender' in col.lower()]
        if gender_cols:
            gender_col = gender_cols[0]
            # Use value_counts for faster unique value retrieval
            gender_values = df[gender_col].dropna().astype(str).str.strip().unique()
            options['genders'] = ['all'] + sorted([v for v in gender_values if v != ''])
        
        # Department options
        dept_cols = [col for col in df.columns if any(word in col.lower() for word in ['department', 'dept', 'branch'])]
        if dept_cols:
            dept_col = dept_cols[0]
            dept_values = df[dept_col].dropna().astype(str).str.strip().unique()
            options['departments'] = ['all'] + sorted([v for v in dept_values if v != ''])
        
        # Co-curricular options
        cocurricular_cols = [col for col in df.columns if any(word in col.lower() for word in ['cocurricular', 'co-curricular', 'extracurricular', 'engaged', 'activity', 'activities'])]
        if cocurricular_cols:
            cocurricular_col = cocurricular_cols[0]
            cocurricular_values = df[cocurricular_col].dropna().astype(str).str.strip().unique()
            options['cocurriculars'] = ['all'] + sorted([v for v in cocurricular_values if v != ''])
        
        # Teacher feedback options
        feedback_cols = [col for col in df.columns if any(word in col.lower() for word in ['teacher_feedback', 'teacher feedback', 'feedback', 'teacher', 'rating'])]
        if feedback_cols:
            feedback_col = feedback_cols[0]
            feedback_values = df[feedback_col].dropna().astype(str).str.strip().unique()
            options['teacher_feedbacks'] = ['all'] + sorted([v for v in feedback_values if v != ''])
            
        # Cache the results
        data_cache.set(cache_key, options)
            
    except Exception as e:
        print(f"❌ Error getting filter options: {e}")
    
    return options

# -------------------------
# Multi-Dataset Credential System - OPTIMIZED
# -------------------------
def load_all_datasets_credentials():
    """Load credentials from ALL available datasets - OPTIMIZED"""
    try:
        all_datasets = get_datasets()
        all_credentials = {}
        
        for dataset_id, dataset_info in all_datasets.items():
            try:
                df, _ = load_data(dataset_id)
                df.columns = [col.strip().lower() for col in df.columns]
                
                id_col = next((col for col in df.columns if 'student' in col and 'id' in col), None)
                pass_col = next((col for col in df.columns if 'password' in col), None)
                
                if id_col and pass_col:
                    # Use vectorized operations for better performance
                    valid_creds = df[[id_col, pass_col]].dropna()
                    for _, row in valid_creds.iterrows():
                        student_id_str = str(row[id_col]).strip()
                        all_credentials[student_id_str] = {
                            'password': str(row[pass_col]).strip(),
                            'dataset_id': dataset_id
                        }
                            
                print(f"✅ Loaded credentials from {dataset_id}")
                
            except Exception as e:
                print(f"⚠️ Error loading {dataset_id}: {e}")
                continue
        
        app.config['all_student_credentials'] = all_credentials
        print(f"🎯 Total students across all datasets: {len(all_credentials)}")
        
    except Exception as e:
        print(f"⚠️ Error in load_all_datasets_credentials: {e}")
        app.config['all_student_credentials'] = {}

@app.before_request
def before_request():
    if 'all_student_credentials' not in app.config:
        load_all_datasets_credentials()

# -------------------------
# COMPREHENSIVE Dashboard - Load ALL data for initial render
# -------------------------
def get_all_plots_data(filtered_df, dataset_id, filters=None):
    """Generate all plots data for comprehensive dashboard view"""
    cache_key = f"all_plots_{dataset_id}_{hash(str(filtered_df.shape) + str(filters))}"
    cached = data_cache.get(cache_key)
    if cached:
        print(f"📦 Loading ALL plots from cache")
        return cached
    
    print(f"🔄 Generating ALL plots data")
    
    plots_data = {
        # Univariate
        "univariate_plots": standardize_plot_format(
            safe_execute(univariate, filtered_df, default={}, 
                       cache_key=generate_cache_key('univariate', filtered_df, filters, dataset_id))
        ),
        "insights": generate_insights(),
        
        # Academic
        "dept_vs_cgpa": safe_execute(dept_vs_cgpa_all, filtered_df, default=[], 
                                   cache_key=generate_cache_key('dept_vs_cgpa_all', filtered_df, filters, dataset_id)),
        "gender_vs_cgpa": safe_execute(gender_vs_cgpa, filtered_df, default=[], 
                                     cache_key=generate_cache_key('gender_vs_cgpa', filtered_df, filters, dataset_id)),
        "academic_insights": generate_insights_Academic(),
        
        # Attendance
        "dept_vs_avg_attendance": safe_execute(dept_vs_avg_attendance, filtered_df, default=[], 
                                             cache_key=generate_cache_key('dept_vs_avg_attendance', filtered_df, filters, dataset_id)),
        "gender_vs_attendance": safe_execute(gender_vs_avg_attendance_all, filtered_df, default=[], 
                                           cache_key=generate_cache_key('gender_vs_avg_attendance_all', filtered_df, filters, dataset_id)),
        "attendance_insights": generate_insights_Attendance(),
        
        # Feedback
        "dept_vs_feedback": safe_execute(dept_vs_feedback_all, filtered_df, default=[], 
                                       cache_key=generate_cache_key('dept_vs_feedback_all', filtered_df, filters, dataset_id)),
        "dept_vs_teacher_feedback": safe_execute(dept_vs_teacher_feedback, filtered_df, default=[], 
                                               cache_key=generate_cache_key('dept_vs_teacher_feedback', filtered_df, filters, dataset_id)),
        "gender_vs_feedback_rating": safe_execute(gender_vs_feedback_rating_all, filtered_df, default=[], 
                                                cache_key=generate_cache_key('gender_vs_feedback_rating_all', filtered_df, filters, dataset_id)),
        "gender_vs_teacher_feedback": safe_execute(gender_vs_teacher_feedback_all, filtered_df, default=[], 
                                                 cache_key=generate_cache_key('gender_vs_teacher_feedback_all', filtered_df, filters, dataset_id)),
        "feedback_insights": generate_insights_Feedback(),
        
        # Engagement
        "dept_vs_co_curricular_all": safe_execute(dept_vs_co_curricular_all, filtered_df, default=[], 
                                                cache_key=generate_cache_key('dept_vs_co_curricular_all', filtered_df, filters, dataset_id)),
        "gender_vs_cocurricular": safe_execute(gender_vs_cocurricular_all, filtered_df, default=[], 
                                             cache_key=generate_cache_key('gender_vs_cocurricular_all', filtered_df, filters, dataset_id)),
        "dept_vs_cocurricular_all": safe_execute(dept_vs_co_curricular_all, filtered_df, default=[], 
                                               cache_key=generate_cache_key('dept_vs_cocurricular_all', filtered_df, filters, dataset_id)),
        "attendance_vs_cocurricular_all": safe_execute(attendance_vs_cocurricular_all, filtered_df, default=[], 
                                                     cache_key=generate_cache_key('attendance_vs_cocurricular_all', filtered_df, filters, dataset_id)),
        "cgpa_vs_cocurricular_all": safe_execute(cgpa_vs_cocurricular_all, filtered_df, default=[], 
                                               cache_key=generate_cache_key('cgpa_vs_cocurricular_all', filtered_df, filters, dataset_id)),
        "engagement_insights": generate_insights_Engagement(),
        
        # Correlation
        "feedback_vs_cgpa_all": safe_execute(feedback_vs_cgpa_all, filtered_df, default=[], 
                                           cache_key=generate_cache_key('feedback_vs_cgpa_all', filtered_df, filters, dataset_id)),
        "attendance_vs_cgpa_all": safe_execute(attendance_vs_cgpa_all, filtered_df, default=[], 
                                             cache_key=generate_cache_key('attendance_vs_cgpa_all', filtered_df, filters, dataset_id)),
        "attendance_vs_feedback_all": safe_execute(attendance_vs_feedback_all, filtered_df, default=[], 
                                                 cache_key=generate_cache_key('attendance_vs_feedback_all', filtered_df, filters, dataset_id)),
        "correlation_insights": generate_insights_Correlation()
    }
    
    data_cache.set(cache_key, plots_data)
    return plots_data

# -------------------------
# Auth routes
# -------------------------
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('user_type'):
        return redirect(url_for('dashboard' if session['user_type'] == 'admin' else 'student'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        user_type = request.form.get('user_type', 'student')

        if not username or not password:
            flash('Please fill all fields.', 'warning')
            return render_template('login.html')

        if user_type == 'admin' and ADMIN_CREDENTIALS.get(username) == password:
            session.update({'user_type': 'admin', 'username': username})
            flash('Admin logged in successfully.', 'success')
            return redirect(url_for('dashboard'))
            
        elif user_type == 'student':
            all_creds = app.config.get('all_student_credentials', {})
            student_info = all_creds.get(username)
            
            if student_info and student_info['password'] == password:
                session.update({
                    'user_type': 'student', 
                    'username': username,
                    'student_dataset': student_info['dataset_id']
                })
                flash('Student logged in successfully.', 'success')
                return redirect(url_for('student'))
            else:
                flash('Invalid Student ID or Password.', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.')
    return redirect(url_for('login'))

# -------------------------
# Dataset routes with cache management
# -------------------------
@app.route('/upload', methods=['POST'])
def upload_file():
    if session.get('user_type') != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('login'))

    if 'file' not in request.files:
        flash('No file selected', 'danger')
        return redirect(url_for('dashboard'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('dashboard'))

    if file and file.filename.lower().endswith('.csv'):
        try:
            dataset_id = add_uploaded_dataset(file, file.filename)
            flash(f'Dataset "{file.filename}" uploaded successfully!', 'success')
            session['current_dataset'] = dataset_id
            session.pop('active_filters', None)
            
            # CLEAR CACHE for this dataset
            data_cache.clear_dataset_cache(dataset_id)
            load_all_datasets_credentials()
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'danger')
    else:
        flash('Please upload a CSV file', 'danger')

    return redirect(url_for('dashboard'))

@app.route('/switch_dataset', methods=['POST'])
def switch_dataset():
    if session.get('user_type') != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('login'))

    dataset_id = request.form.get('dataset_select', 'default')

    try:
        all_datasets = get_datasets()
        if dataset_id not in all_datasets:
            flash('Invalid dataset selected.', 'danger')
        else:
            session['current_dataset'] = dataset_id
            session.pop('active_filters', None)
            flash(f"✅ Switched to dataset: {all_datasets[dataset_id]['name']}", 'success')
    except Exception as e:
        flash(f'Error switching dataset: {e}', 'danger')

    return redirect(url_for('dashboard'))

# -------------------------
# OPTIMIZED Dashboard route - FIXED: All sections load plots upfront with proper filtering
# -------------------------
@app.route('/dashboard')
def dashboard():
    if session.get('user_type') != 'admin':
        flash('Access denied. Please log in as Admin.')
        return redirect(url_for('login'))

    current_dataset = session.get('current_dataset', 'default')
    start_time = time.time()
    
    try:
        # Load dataframe and dataset metadata
        df, dataset_info = load_data(current_dataset)
        
        # Get section from URL parameters, default to univariate
        section = request.args.get('section', 'univariate')
        
        # Get filters from URL parameters
        filters = {
            'gender': request.args.get('gender', 'all'),
            'department': request.args.get('department', 'all'),
            'cocurricular': request.args.get('cocurricular', 'all'),
            
        }

        filtered_df = apply_filters(df, filters)

        # ✅ CRITICAL FIX: For Univariate section, use original unfiltered data
        # For all other sections, use filtered data
        if section == 'univariate':
            # Univariate always shows complete data (no filters)
            display_df = df
            filtered_rows_display = len(df)  # Show total rows for univariate
            print(f"📊 Univariate section - Using UNFILTERED data ({len(df)} records)")
        else:
            # Other sections use filtered data
            display_df = filtered_df
            filtered_rows_display = len(display_df)
            print(f"📊 {section} section - Using FILTERED data ({filtered_rows_display} records)")
        
        # Get filter options for dropdowns
        filter_options = get_filter_options(df)

        # ✅ CRITICAL FIX: Always use get_all_plots_data to load ALL plots upfront
        # This ensures all sections have their plots ready when switching
        plots_data = get_all_plots_data(display_df, current_dataset, filters)

        # Base plots data structure
        plots_data.update({
            "dataset_info": {
                "name": dataset_info.get("name", "Dataset"),
                "rows": dataset_info.get("rows", len(df)),
                "filtered_rows": filtered_rows_display,  # Use appropriate row count
                "columns": dataset_info.get("columns", len(df.columns))
            },
            "filters": filters,
            "filter_options": filter_options,
            "active_section": section,
            "show_filters": section != 'univariate'  # Hide filters for univariate
        })

        all_datasets = get_datasets()
        
        
        return render_template(
            'dashboard.html',
            username=session.get('username'),
            plots=plots_data,
            df_columns=df.columns.tolist(),
            datasets=all_datasets,
            current_dataset=current_dataset
        )
        
    except Exception as e:
        flash(f"Error generating dashboard: {str(e)}", "danger")
        print(f"❌ Dashboard error: {e}")
        import traceback
        traceback.print_exc()
        
        # Create a comprehensive plots structure to prevent template errors
        empty_plots = {
            "dataset_info": {"name": "N/A", "rows": 0, "filtered_rows": 0, "columns": 0}, 
            "univariate_plots": {},
            "dept_vs_cgpa": [],
            "gender_vs_cgpa": [],
            "dept_vs_avg_attendance": [],
            "gender_vs_attendance": [],
            "dept_vs_feedback": [],
            "dept_vs_teacher_feedback": [],
            "gender_vs_feedback_rating": [],
            "gender_vs_teacher_feedback": [],
            "dept_vs_co_curricular_all": [],
            "gender_vs_cocurricular": [],
            "dept_vs_cocurricular_all": [],
            "attendance_vs_cocurricular_all": [],
            "cgpa_vs_cocurricular_all": [],
            "feedback_vs_cgpa_all": [],
            "attendance_vs_cgpa_all": [],
            "attendance_vs_feedback_all": [],
            "insights": {"title": "No Data Available", "content": "<p>Unable to load dataset insights.</p>"},
            "academic_insights": {"title": "No Data Available", "content": "<p>Unable to load academic insights.</p>"},
            "attendance_insights": {"title": "No Data Available", "content": "<p>Unable to load attendance insights.</p>"},
            "feedback_insights": {"title": "No Data Available", "content": "<p>Unable to load feedback insights.</p>"},
            "engagement_insights": {"title": "No Data Available", "content": "<p>Unable to load engagement insights.</p>"},
            "correlation_insights": {"title": "No Data Available", "content": "<p>Unable to load correlation insights.</p>"},
            "filters": {},
            "filter_options": {"genders": ["all"], "departments": ["all"], "cocurriculars": ["all"], "teacher_feedbacks": ["all"]},
            "active_section": section if 'section' in locals() else "univariate",
            "show_filters": section != 'univariate' if 'section' in locals() else False
        }
        
        try:
            all_datasets = get_datasets()
        except Exception:
            all_datasets = {"default": {"name": "Default Dataset", "path": "default/data.csv", "is_default": True}}

        return render_template(
            'dashboard.html',
            username=session.get('username'),
            plots=empty_plots,
            df_columns=[],
            datasets=all_datasets,
            current_dataset=current_dataset
        )

# ---------- STUDENT PAGE ----------
@app.route("/student")
def student():
    if session.get('user_type') != 'student':
        flash('Access denied. Please log in as Student.')
        return redirect(url_for('login'))

    student_id = session['username']
    student_dataset = session.get('student_dataset', 'default')
    
    try:
        # Load from student's specific dataset
        df, _ = load_data(student_dataset)
        df.columns = [col.strip().lower() for col in df.columns]

        # Find student ID and name columns
        id_col = next((col for col in df.columns if 'student' in col and 'id' in col), None)
        name_col = next((col for col in df.columns if 'name' in col), None)

        student_data = {}
        student_name = "Student"

        if id_col:
            student_row = df[df[id_col].astype(str).str.strip() == student_id.strip()]
            if not student_row.empty:
                student_data = student_row.to_dict(orient='records')[0]

                # Get student name
                if name_col and name_col in student_data and pd.notna(student_data[name_col]):
                    student_name = student_data[name_col]
                else:
                    for col in ['name', 'student_name', 'studentname']:
                        if col in student_data and pd.notna(student_data[col]):
                            student_name = student_data[col]
                            break

                # Remove password & name from display
                student_data = {
                    k: v for k, v in student_data.items()
                    if 'password' not in k.lower() and 'name' not in k.lower()
                }

        return render_template(
            "student.html",
            username=student_name,
            student_data=student_data,
            student_id=student_id,
            dataset_name=student_dataset
        )
        
    except Exception as e:
        flash(f'Error loading student data from {student_dataset}: {str(e)}', 'danger')
        return render_template(
            "student.html",
            username="Student",
            student_data={},
            student_id=student_id
        )

if __name__ == "__main__":
    print("🚀 Starting optimized Flask application...")
    print("📊 Performance features enabled:")
    print("   ✅ Advanced caching system")
    print("   ✅ Univariate independent from filters")
    print("   ✅ All sections pre-loaded")
    print("   ✅ Filter persistence across sections")
    app.run(debug=True)