import json
import subprocess
import threading
from io import BytesIO
from flask import Flask, send_file, request, jsonify, current_app, session
from PIL import Image
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask_bcrypt import Bcrypt
from flask_session import Session
import logging
import base64
import os
from functools import wraps
import db
import main
from werkzeug.security import generate_password_hash, check_password_hash
import hashlib
from main import *


def load_config(filename):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config


def create_app(config_name):
    app = Flask(__name__)
    bcrypt = Bcrypt()
    CORS(app, supports_credentials=True)

    config = load_config('config.json')

    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SECRET_KEY'] = 'secret123'

    if config_name == 'production':
        app.logger.setLevel(logging.INFO)
    else:
        app.logger.setLevel(logging.DEBUG)

    Session(app)

    conn = db.connect_to_database(config['db_host'], config['db_user'], config['db_password'], config['db'])

    def login_required(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            return f(*args, **kwargs)

        return decorated_function

    def str_to_bool(s):
        if isinstance(s, bool):
            return s
        elif s == 'true':
            return True
        elif s == 'false':
            return False
        else:
            raise ValueError("Input must be 'true', 'false' or a boolean")

    @app.route("/txt2img", methods=["POST"])
    @login_required
    def txt2img():
        prompt = request.form.get('prompt')
        negative = request.form.get('negative', '')
        guidance_scale = float(request.form.get('guidance_scale', 10))
        personalized = str_to_bool(request.form.get('personalized', True))

        # app.logger.info(f"personalized: {personalized}")
        # if isinstance(personalized, str):
        #     app.logger.info(f"personalized is a string")

        model = request.form.get('model')

        if model not in main.get_available_generic_models() and model != "Lightning":
            return jsonify({'error': 'Invalid model provided'}), 400

        refiner = request.form.get('refiner', 'none')

        if refiner not in main.get_available_refiners() and refiner != "none":
            return jsonify({'error': 'Invalid refiner provided'}), 400

        if not prompt or prompt=='':
            return jsonify({'error': 'No prompt provided'}), 400

        username = db.get_username_by_id(conn, session['user_id'])

        img_path = main.txt2img(prompt, negative, guidance_scale, model, refiner=refiner, username=username,
                                personalized=personalized)

        with open(img_path, 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        img_name = img_path.split('/')[-1]

        return jsonify({'image_name': img_name, 'image': img_base64})

    def allowed_file(filename):
        ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route("/get_models", methods=["GET"])
    def get_models():
        generic_models = main.get_available_generic_models()
        refiners = main.get_available_refiners()
        refiners.insert(0, "none")

        return jsonify({'generic_models': generic_models, 'refiners': refiners})

    @app.route("/img2img", methods=["POST"])
    @login_required
    def img2img():
        if 'image' not in request.files:
            print('No image part')
            return jsonify({'error': 'No image part'}), 400

        file = request.files['image']

        if file.filename == '':
            print('No selected file')
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            image = Image.open(BytesIO(file.read()))
            if image.mode in ('L', '1', 'I;16'):
                image = image.convert('RGB')

        negative = request.form.get('negative', '')
        prompt = request.form.get('prompt')
        guidance_scale = float(request.form.get('guidance_scale', 10))
        strength = float(request.form.get('strength', 0.5))
        personalized = str_to_bool(request.form.get('personalized', True))

        if not prompt or prompt == '':
            return jsonify({'error': 'No prompt provided'}), 400

        model = request.form.get('model')

        if model not in main.get_available_generic_models():
            return jsonify({'error': 'Invalid model provided'}), 400

        refiner = request.form.get('refiner', 'none')

        if refiner not in main.get_available_refiners() and refiner != "none":
            return jsonify({'error': 'Invalid refiner provided'}), 400

        username = db.get_username_by_id(conn, session['user_id'])

        img_paths = main.img2img(prompt, negative, image, guidance_scale, strength, model, refiner=refiner,
                                 username=username, personalized=personalized)

        with open(img_paths[0], 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        img_name = img_paths[0].split('/')[-1]

        return jsonify({'image_name': img_name, 'image': img_base64})

    @app.route("/gallery", methods=["GET"])
    @login_required
    def gallery():
        ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
        gallery_path = "./users/" + db.get_username_by_id(conn, session['user_id']) + "/gallery/"
        files = os.listdir(gallery_path)
        images = []
        for file in files:
            if file.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
                with open(gallery_path + file, 'rb') as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                images.append({'image_name': file, 'image': img_base64})
        images.reverse()
        return jsonify(images)

    @app.route("/save_to_gallery", methods=["POST"])
    @login_required
    def save_to_gallery():
        image_base64 = request.form.get('image')
        image_name = request.form.get('image_name')

        gallery_path = "./users/" + db.get_username_by_id(conn, session['user_id']) + "/gallery/"

        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400

        if not image_name:
            return jsonify({'error': 'No image name provided'}), 400

        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))

            image_path = os.path.join(gallery_path, secure_filename(image_name))
            image.save(image_path)

            return jsonify({'message': 'Image uploaded successfully'}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    @app.route("/delete_from_gallery", methods=["POST"])
    @login_required
    def delete_from_galley():
        image_name = request.form.get('image_name')
        gallery_path = "./users/" + db.get_username_by_id(conn, session['user_id']) + "/gallery/"
        try:
            os.remove(gallery_path + image_name)
            return jsonify({'message': 'Image deleted successfully'}), 200
        except Exception as e:
            return jsonify({'error': "Image not found."}), 400

    def ensure_rgb(image):
        if image.mode != 'RGB':
            return image.convert('RGB')
        return image

    @app.route("/rgb", methods=["POST"])
    @login_required
    def rgb():
        images = []
        i = 0
        while True:
            image_key = 'image' + str(i)
            if image_key not in request.files:
                break
            file = request.files[image_key]
            if file and allowed_file(file.filename):
                image = Image.open(BytesIO(file.read()))
                image = ensure_rgb(image)
                images.append(image)
            i += 1

        if not images:
            return jsonify({'error': 'No images received'}), 400

        rgb_image = create_rgb_image(images)

        buffered = BytesIO()
        rgb_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        current_datetime = datetime.now()
        current_datetime_str = current_datetime.strftime("%Y%m%d_%H%M%S")

        return jsonify({'message': f'Received {len(images)} images', 'image': img_str,
                        'image_name': current_datetime_str + ".jpg"}), 200

    @app.route("/merge", methods=["POST"])
    @login_required
    def merge():
        images = []
        i = 0
        while True:
            image_key = 'image' + str(i)
            if image_key not in request.files:
                break
            file = request.files[image_key]
            if file and allowed_file(file.filename):
                image = Image.open(BytesIO(file.read()))
                image = ensure_rgb(image)
                images.append(image)
            i += 1

        if not images:
            return jsonify({'error': 'No images received'}), 400

        merged_image = merge_images(images)

        buffered = BytesIO()
        merged_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        current_datetime = datetime.now()
        current_datetime_str = current_datetime.strftime("%Y%m%d_%H%M%S")

        return jsonify({'message': f'Received {len(images)} images', 'image': img_str,
                        'image_name': current_datetime_str + ".jpg"}), 200

    @app.route("/merge_gray", methods=["POST"])
    @login_required
    def merge_gray():
        images = []
        i = 0
        while True:
            image_key = 'image' + str(i)
            if image_key not in request.files:
                break
            file = request.files[image_key]
            if file and allowed_file(file.filename):
                image = Image.open(BytesIO(file.read()))
                image = ensure_rgb(image)
                images.append(image)
            i += 1

        if not images:
            return jsonify({'error': 'No images received'}), 400

        merged_image = merge_images_gray(images)

        buffered = BytesIO()
        merged_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        current_datetime = datetime.now()
        current_datetime_str = current_datetime.strftime("%Y%m%d_%H%M%S")

        return jsonify({'message': f'Received {len(images)} images', 'image': img_str,
                        'image_name': current_datetime_str + ".jpg"}), 200

    @app.route('/register', methods=['POST'])
    def register():
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not all([username, email, password]):
            return jsonify({'error': 'All fields must be filled.'}), 400

        user_exists = db.user_exists(conn, username, email)

        if user_exists[0]:
            return jsonify({'error': user_exists[1]}), 400

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        db.create_user(conn, username, hashed_password, email)

        user_directory = os.path.join('users', username)
        os.makedirs(user_directory)

        gallery_directory = os.path.join(user_directory, 'gallery')
        train_directory = os.path.join(user_directory, 'train')
        images_directory = os.path.join(user_directory, 'images')
        os.makedirs(gallery_directory)
        os.makedirs(train_directory)
        os.makedirs(images_directory)
        train_img_directory = os.path.join(train_directory, 'images')
        train_cap_directory = os.path.join(train_directory, 'captions')
        os.makedirs(train_img_directory)
        os.makedirs(train_cap_directory)

        return jsonify({'message': 'User registered successfully'}), 201

    @app.route('/login', methods=['POST'])
    def login():
        email = request.form.get('email')
        password = request.form.get('password')

        if not all([email, password]):
            return jsonify({'error': 'All fields must be filled.'}), 400

        user = db.get_user_by_email(conn, email)
        if not user:
            return jsonify({'error': 'Invalid email or password'}), 401

        if bcrypt.check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            response = jsonify({'message': 'Login successful'})
            response.status_code = 200
            return response
        else:
            return jsonify({'error': 'Invalid email or password'}), 401

    @app.route('/logout', methods=['POST'])
    @login_required
    def logout():
        session.pop('user_id', None)
        return jsonify({'message': 'Logged out successfully'}), 200

    @app.route('/check-session', methods=['GET'])
    def check_session():
        if 'user_id' in session:
            return jsonify(loggedIn=True)
        else:
            return jsonify(loggedIn=False)

    def read_output(pipe, logger_func, is_error=False):
        for line in iter(pipe.readline, ''):
            if is_error:
                logger_func(f"ERROR: Thread {threading.current_thread().name}: {line}")
            else:
                logger_func(f"Thread {threading.current_thread().name}: {line}")

    def run_training(username):

        python_executable = './venv/Scripts/python.exe'
        command = [python_executable, "training.py", "--username", username]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, app.logger.info, False),
                                         name="stdout_thread")
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, app.logger.info, True),
                                         name="stderr_thread")

        stdout_thread.start()
        stderr_thread.start()

        process.wait()

        stdout_thread.join()
        stderr_thread.join()

        return process.returncode

    @app.route('/personalize', methods=['POST'])
    @login_required
    def personalize():
        username = db.get_username_by_id(conn, session['user_id'])

        gallery_path = "./users/" + username + "/gallery/"
        files = os.listdir(gallery_path)
        if len(files) == 0:
            return jsonify({'error': 'No images in gallery'}), 400

        rcode = run_training(username)
        if rcode != 0:
            return jsonify({'error': 'Error occurred during training'}), 400
        return jsonify({'message': 'Personalization done.'}), 200

    return app
