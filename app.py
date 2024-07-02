from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
from flask_login import LoginManager, login_user, login_required, logout_user
from forms import LoginForm, RegisterForm
from models import User, users  # Ensure this import points to where you define User and users
from data_processing import get_text_from_html_doc
from new_predict import PatentPredicterAI
from listes_labels import list_label_level_0
from werkzeug.security import check_password_hash, generate_password_hash
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'killian'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.get(form.username.data)
        if user and users.get(user.id, {}).get('password') == form.password.data:
            login_user(user)
            return redirect(url_for('index'))  # Redirige vers la page index après la connexion
        else:
            flash('Invalid username or password')
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        if form.username.data not in users:
            users[form.username.data] = {'password': form.password.data}
            logging.debug(f"User {form.username.data} registered with password {form.password.data}")
            return redirect(url_for('login'))
        flash('Username already exists')
    return render_template('register.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/about')
@login_required
def about():
    return render_template('aPropos.html')

@app.route('/faq')
@login_required
def faq():
    return render_template('faq.html')

@app.route('/contact')
@login_required
def contact():
    return render_template('contact.html')

try:
    predictor = PatentPredicterAI(level=0)
    logging.debug("Predictor initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize the predictor: {str(e)}")

@app.route('/classify', methods=['POST'])
@login_required
def classify():
    try:
        data = request.get_json()
        raw_text = data['input_data']
        prediction_type = data['type']  # Retrieves the type of prediction

        if prediction_type == 'quick':
            predictions = predictor.quick_predict(raw_text)
            return jsonify({'predictions': predictions}), 200
        else:
            clean_text, predictions, attributions = predictor.full_predict(raw_text)
            formatted_attributions = {word: float(score) for word, score in attributions}
            return jsonify({'clean_text': clean_text, 'predictions': predictions, 'attributions': formatted_attributions}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
