# Text Analysis Platform

A web application that allows users to upload and compare text files using various algorithms including word frequency, cosine similarity, and Levenshtein distance.

## Features

- User authentication (register/login)
- Admin panel for user management
- Multiple file upload support
- Text file comparison using:
  - Word frequency analysis
  - Cosine similarity
  - Levenshtein distance
- Modern and responsive UI

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Access the application at `http://localhost:5000`

## Usage

1. Register a new account or login with existing credentials
2. Upload one or more text files using the file upload form
3. View the comparison results showing:
   - Cosine similarity between files
   - Levenshtein distance similarity
   - Word frequency analysis

## Admin Features

- Access the admin panel at `/admin`
- View all registered users
- Toggle admin status for users

## Security Notes

- Change the `SECRET_KEY` in `app.py` before deploying to production
- Implement proper password policies in production
- Use HTTPS in production
- Implement rate limiting for file uploads 