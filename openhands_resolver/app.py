import os
import requests
from flask import Flask, request, jsonify
import jwt
import time
from functools import wraps
import hmac
import hashlib
import json
from .resolve_issue import resolve_issue
from .send_pull_request import send_pull_request

app = Flask(__name__)

def verify_webhook_signature(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        webhook_secret = os.environ.get('GITHUB_WEBHOOK_SECRET')
        if not webhook_secret:
            return jsonify({'error': 'GITHUB_WEBHOOK_SECRET not configured'}), 500

        signature = request.headers.get('X-Hub-Signature-256')
        if not signature:
            return jsonify({'error': 'No signature header found'}), 400

        hash_object = hmac.new(
            webhook_secret.encode('utf-8'),
            msg=request.get_data(),
            digestmod=hashlib.sha256
        )
        expected_signature = f"sha256={hash_object.hexdigest()}"

        if not hmac.compare_digest(signature, expected_signature):
            return jsonify({'error': 'Invalid signature'}), 401

        return f(*args, **kwargs)
    return decorated_function

def get_jwt_token():
    app_id = os.environ.get('GITHUB_APP_ID')
    private_key = os.environ.get('GITHUB_PRIVATE_KEY')
    
    if not app_id or not private_key:
        raise ValueError("GITHUB_APP_ID and GITHUB_PRIVATE_KEY must be set")
    
    now = int(time.time())
    payload = {
        'iat': now,
        'exp': now + (10 * 60),  # 10 minutes expiration
        'iss': app_id
    }
    
    return jwt.encode(payload, private_key, algorithm='RS256')

def get_installation_token(installation_id):
    jwt_token = get_jwt_token()
    headers = {
        'Authorization': f'Bearer {jwt_token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    response = requests.post(
        f'https://api.github.com/app/installations/{installation_id}/access_tokens',
        headers=headers
    )
    response.raise_for_status()
    return response.json()['token']

@app.route('/webhook', methods=['POST'])
@verify_webhook_signature
def webhook():
    event = request.headers.get('X-GitHub-Event')
    payload = request.json
    
    if event not in ['issues', 'pull_request']:
        return jsonify({'status': 'ignored', 'reason': f'Event {event} not handled'}), 200
        
    # Get installation token
    installation_id = payload['installation']['id']
    token = get_installation_token(installation_id)
    os.environ['GITHUB_TOKEN'] = token
    
    try:
        if event == 'issues' and payload['action'] == 'labeled':
            label = payload['label']['name']
            if label == 'fix-me':
                issue_number = payload['issue']['number']
                repo = payload['repository']['full_name']
                
                # Attempt to resolve the issue
                resolve_issue(repo=repo, issue_number=issue_number)
                
                # Check if resolution was successful
                with open('output/output.jsonl', 'r') as f:
                    last_line = f.readlines()[-1]
                    result = json.loads(last_line)
                    if result.get('success'):
                        # Create draft PR
                        send_pull_request(issue_number=issue_number, pr_type='draft')
                    else:
                        # Push branch only
                        send_pull_request(issue_number=issue_number, pr_type='branch', send_on_failure=True)
                        
        elif event == 'pull_request' and payload['action'] == 'labeled':
            label = payload['label']['name']
            if label == 'fix-me':
                pr_number = payload['pull_request']['number']
                repo = payload['repository']['full_name']
                
                # Attempt to resolve the PR
                resolve_issue(repo=repo, issue_number=pr_number, issue_type='pr')
                
                # Always push changes to the PR branch
                send_pull_request(issue_number=pr_number, pr_type='branch', send_on_failure=True)
                
        return jsonify({'status': 'success'}), 200
        
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)