from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, make_response, abort
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from datetime import datetime, timedelta
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from Levenshtein import distance
import json
from sqlalchemy import func
import zipfile
from io import BytesIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'our_best_key'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024  # 16GB max file size
app.config['FREE_SCANS_PER_DAY'] = 20  # Number of free scans per day

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    credits = db.Column(db.Integer, default=0)
    daily_scans_used = db.Column(db.Integer, default=0)
    last_scan_date = db.Column(db.Date, default=datetime.utcnow().date)
    analyses = db.relationship('Analysis', backref='user', lazy=True)

    @property
    def remaining_free_scans(self):
        today = datetime.utcnow().date()
        if self.last_scan_date != today:
            self.daily_scans_used = 0
            self.last_scan_date = today
            db.session.commit()
        
        remaining = app.config['FREE_SCANS_PER_DAY'] - self.daily_scans_used
        return max(0, remaining)

    @property
    def total_remaining_scans(self):
        return self.remaining_free_scans + self.credits

    def has_credits(self):
        return self.total_remaining_scans > 0

    def use_credit(self):
        if self.remaining_free_scans > 0:
            self.daily_scans_used += 1
        else:
            self.credits -= 1
        db.session.commit()

    def __repr__(self):
        return f'<User {self.username}>'

class CreditRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    amount = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, approved, rejected
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    processed_at = db.Column(db.DateTime)
    
    # Add relationship to User
    user = db.relationship('User', backref=db.backref('credit_requests', lazy=True))

class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    file_count = db.Column(db.Integer, default=0)

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    files = db.Column(db.String(500), nullable=False)
    word_frequency = db.Column(db.Text, nullable=False)
    cosine_similarity = db.Column(db.Float, nullable=False)
    levenshtein_similarity = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f'<Analysis {self.id}>'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def word_frequency(text):
    # Convert to lowercase and remove punctuation
    text = text.lower()
    for char in '.,!?;:()[]{}\'"':
        text = text.replace(char, ' ')
    
    # Split into words and count frequencies
    words = text.split()
    freq = {}
    for word in words:
        if word:  # Skip empty strings
            freq[word] = freq.get(word, 0) + 1
    
    # Sort by frequency
    sorted_freq = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
    return sorted_freq

def cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = (tfidf_matrix * tfidf_matrix.T).toarray()[0][1]
        return max(0.0, min(1.0, similarity))  # Ensure value is between 0 and 1
    except Exception as e:
        print(f"Error in cosine similarity: {e}")
        return 0.0

def levenshtein_similarity(text1, text2):
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 1.0
    return 1 - (distance(text1, text2) / max_len)

@app.route('/')
def index():
    return render_template('landing.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists')
            return redirect(url_for('register'))
        
        user = User(username=username, email=email, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        
        flash('Invalid email or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get recent analyses
    recent_analyses = Analysis.query.filter_by(user_id=current_user.id)\
        .order_by(Analysis.timestamp.desc())\
        .limit(5)\
        .all()
    
    # Process analyses to include mean similarity
    for analysis in recent_analyses:
        analysis.files = json.loads(analysis.files)
        # Calculate mean similarity
        analysis.mean_similarity = (analysis.cosine_similarity + analysis.levenshtein_similarity) / 2
    
    return render_template('dashboard.html',
                         remaining_free_scans=current_user.remaining_free_scans,
                         credits=current_user.credits,
                         total_remaining_scans=current_user.total_remaining_scans,
                         recent_analyses=recent_analyses)

@app.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        return redirect(url_for('dashboard'))
    
    # Get statistics
    total_scans = Scan.query.count()
    scans_today = Scan.query.filter(
        func.date(Scan.created_at) == datetime.utcnow().date()
    ).count()
    
    # Get top users with their activity
    top_users = db.session.query(
        User,
        func.count(Scan.id).label('total_scans'),
        func.count(Analysis.id).label('total_analyses'),
        func.max(Analysis.timestamp).label('last_activity')
    ).outerjoin(Scan, User.id == Scan.user_id)\
     .outerjoin(Analysis, User.id == Analysis.user_id)\
     .group_by(User.id)\
     .order_by(func.count(Scan.id).desc())\
     .limit(10)\
     .all()
    
    # Get pending credit requests
    pending_requests = CreditRequest.query.filter_by(status='pending').all()
    
    return render_template('admin.html',
                         users=User.query.all(),
                         total_scans=total_scans,
                         scans_today=scans_today,
                         top_users=top_users,
                         pending_requests=pending_requests)

@app.route('/request-credit', methods=['POST'])
@login_required
def request_credit():
    amount = request.form.get('amount', type=int)
    if amount and amount > 0:
        credit_request = CreditRequest(user_id=current_user.id, amount=amount)
        db.session.add(credit_request)
        db.session.commit()
        flash('Credit request submitted successfully')
    return redirect(url_for('dashboard'))

@app.route('/admin/process-credit-request', methods=['POST'])
@login_required
def process_credit_request():
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    request_id = request.form.get('request_id', type=int)
    action = request.form.get('action')
    
    credit_request = CreditRequest.query.get_or_404(request_id)
    if action in ['approve', 'reject']:
        credit_request.status = 'approved' if action == 'approve' else 'rejected'
        credit_request.processed_at = datetime.utcnow()
        
        # If approved, add credits to user
        if action == 'approve':
            credit_request.user.credits += credit_request.amount
        
        db.session.commit()
        return jsonify({'success': True})
    
    return jsonify({'error': 'Invalid action'}), 400

@app.route('/upload_files', methods=['POST'])
@login_required
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files[]')
    if len(files) < 2:
        return jsonify({'error': 'Please upload at least two files'}), 400
    
    if not current_user.has_credits():
        return jsonify({'error': 'No credits remaining'}), 403
    
    # Ensure upload directory exists
    upload_dir = os.path.abspath(app.config['UPLOAD_FOLDER'])
    os.makedirs(upload_dir, exist_ok=True)
    print(f"Upload directory: {upload_dir}")  # Debug print
    
    results = []
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            file1 = files[i]
            file2 = files[j]
            
            if file1.filename.endswith('.txt') and file2.filename.endswith('.txt'):
                # Save files to upload directory
                file1_name = secure_filename(file1.filename)
                file2_name = secure_filename(file2.filename)
                
                file1_path = os.path.join(upload_dir, file1_name)
                file2_path = os.path.join(upload_dir, file2_name)
                
                print(f"Saving file 1 to: {file1_path}")  # Debug print
                print(f"Saving file 2 to: {file2_path}")  # Debug print
                
                file1.save(file1_path)
                file2.save(file2_path)
                
                # Verify files were saved
                if not os.path.exists(file1_path) or not os.path.exists(file2_path):
                    print(f"Error: Files were not saved correctly")  # Debug print
                    return jsonify({'error': 'Error saving files'}), 500
                
                # Read file contents
                with open(file1_path, 'r', encoding='utf-8') as f:
                    text1 = f.read()
                with open(file2_path, 'r', encoding='utf-8') as f:
                    text2 = f.read()
                
                # Perform analysis
                word_freq1 = word_frequency(text1)
                word_freq2 = word_frequency(text2)
                cosine_sim = cosine_similarity(text1, text2)
                levenshtein_sim = levenshtein_similarity(text1, text2)
                
                # Store analysis in database
                analysis = Analysis(
                    user_id=current_user.id,
                    files=json.dumps([file1_name, file2_name]),
                    word_frequency=json.dumps(word_freq1),
                    cosine_similarity=float(cosine_sim),
                    levenshtein_similarity=float(levenshtein_sim)
                )
                db.session.add(analysis)
                
                results.append({
                    'files': [file1_name, file2_name],
                    'word_frequency': word_freq1,
                    'cosine_similarity': float(cosine_sim),
                    'levenshtein_similarity': float(levenshtein_sim)
                })
    
    db.session.commit()
    current_user.use_credit()
    db.session.commit()
    
    return jsonify(results)

@app.route('/history')
@login_required
def history():
    analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.timestamp.desc()).all()
    for analysis in analyses:
        analysis.files = json.loads(analysis.files)
        analysis.word_frequency = json.loads(analysis.word_frequency)
        # Calculate if documents are similar based on both metrics
        analysis.is_similar = (analysis.cosine_similarity >= 0.7 or analysis.levenshtein_similarity >= 0.7)
    return render_template('history.html', analyses=analyses)

@app.route('/download_analysis/<int:analysis_id>')
@login_required
def download_analysis(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    if analysis.user_id != current_user.id:
        abort(403)
    
    # Create a JSON file with the analysis data
    data = {
        'files': json.loads(analysis.files),
        'word_frequency': json.loads(analysis.word_frequency),
        'cosine_similarity': analysis.cosine_similarity,
        'levenshtein_similarity': analysis.levenshtein_similarity,
        'timestamp': analysis.timestamp.isoformat()
    }
    
    response = make_response(json.dumps(data, indent=2))
    response.headers['Content-Disposition'] = f'attachment; filename=analysis_{analysis_id}.json'
    response.headers['Content-Type'] = 'application/json'
    return response

@app.route('/admin/add-credits', methods=['POST'])
@login_required
def add_credits():
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    user_id = request.form.get('user_id', type=int)
    amount = request.form.get('amount', type=int)
    
    if not user_id or not amount or amount <= 0:
        return jsonify({'error': 'Invalid parameters'}), 400
    
    user = User.query.get_or_404(user_id)
    user.credits += amount
    
    db.session.commit()
    return jsonify({'success': True})

@app.route('/admin/toggle-admin', methods=['POST'])
@login_required
def toggle_admin():
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    user_id = request.form.get('user_id', type=int)
    if not user_id:
        return jsonify({'error': 'Invalid user ID'}), 400
    
    user = User.query.get_or_404(user_id)
    
    # Don't allow toggling your own admin status
    if user.id == current_user.id:
        return jsonify({'error': 'Cannot change your own admin status'}), 400
    
    user.is_admin = not user.is_admin
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/download_similar/<int:analysis_id>')
@login_required
def download_similar(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    if analysis.user_id != current_user.id:
        abort(403)
    
    # Check if documents are similar
    if analysis.cosine_similarity >= 0.7 or analysis.levenshtein_similarity >= 0.7:
        try:
            # Create a zip file in memory
            memory_file = BytesIO()
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Get the list of files from the analysis
                files = json.loads(analysis.files)
                
                # Add each file to the zip
                for file_name in files:
                    # Get the absolute path to the uploads directory
                    upload_dir = os.path.abspath(app.config['UPLOAD_FOLDER'])
                    file_path = os.path.join(upload_dir, file_name)
                    
                    print(f"Looking for file at: {file_path}")  # Debug print
                    
                    if os.path.exists(file_path):
                        # Add file to zip with its original name
                        zf.write(file_path, os.path.basename(file_name))
                    else:
                        print(f"File not found: {file_path}")  # Debug print
                        flash(f'File not found: {file_name}')
                        return redirect(url_for('history'))
            
            # Prepare the response
            memory_file.seek(0)
            response = make_response(memory_file.read())
            response.headers['Content-Disposition'] = f'attachment; filename=similar_documents_{analysis_id}.zip'
            response.headers['Content-Type'] = 'application/zip'
            return response
        except Exception as e:
            print(f"Error creating zip file: {str(e)}")  # Debug print
            flash(f'Error creating zip file: {str(e)}')
            return redirect(url_for('history'))
    else:
        flash('Documents do not meet the similarity threshold of 70%')
        return redirect(url_for('history'))

@app.route('/admin/delete-user', methods=['POST'])
@login_required
def delete_user():
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    user_id = request.form.get('user_id', type=int)
    if not user_id:
        return jsonify({'error': 'Invalid user ID'}), 400
    
    user = User.query.get_or_404(user_id)
    
    # Don't allow deleting your own account
    if user.id == current_user.id:
        return jsonify({'error': 'Cannot delete your own account'}), 400
    
    try:
        # Delete all associated data
        Analysis.query.filter_by(user_id=user_id).delete()
        Scan.query.filter_by(user_id=user_id).delete()
        CreditRequest.query.filter_by(user_id=user_id).delete()
        
        # Delete the user
        db.session.delete(user)
        db.session.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Create database tables
with app.app_context():
    # Only create tables if they don't exist
    db.create_all()
    
    # Create admin user if it doesn't exist
    if not User.query.filter_by(email='admin@gmail.com').first():
        admin = User(
            username='admin',
            email='admin@gmail.com',
            password_hash=generate_password_hash('admin123'),
            is_admin=True,
            credits=0,
            daily_scans_used=0
        )
        db.session.add(admin)
        db.session.commit()
        print("Admin user created with email: admin@gmail.com, password: admin123")

if __name__ == '__main__':
    app.run(port=5001)  # or any other available port 