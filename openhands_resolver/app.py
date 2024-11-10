import os
import requests
from fastapi import FastAPI, Request, HTTPException, Depends
import jwt
import time
import hmac
import hashlib
from .resolve_issue import resolve_issue
from .send_pull_request import send_pull_request
from openhands.core.config import LLMConfig
from .resolver_output import ResolverOutput

app = FastAPI()

async def verify_webhook_signature(request: Request):
    webhook_secret = os.environ.get('GITHUB_WEBHOOK_SECRET')
    if not webhook_secret:
        raise HTTPException(status_code=500, detail='GITHUB_WEBHOOK_SECRET not configured')

    signature = request.headers.get('X-Hub-Signature-256')
    if not signature:
        raise HTTPException(status_code=400, detail='No signature header found')

    body = await request.body()
    hash_object = hmac.new(
        webhook_secret.encode('utf-8'),
        msg=body,
        digestmod=hashlib.sha256
    )
    expected_signature = f"sha256={hash_object.hexdigest()}"

    if not hmac.compare_digest(signature, expected_signature):
        raise HTTPException(status_code=401, detail='Invalid signature')

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

@app.post('/webhook')
async def webhook(request: Request, _: None = Depends(verify_webhook_signature)):
    event = request.headers.get('X-GitHub-Event')
    payload = await request.json()
    
    if event not in ['issues', 'pull_request']:
        return {'status': 'ignored', 'reason': f'Event {event} not handled'}
        
    # Get installation token
    installation_id = payload['installation']['id']
    token = get_installation_token(installation_id)
    os.environ['GITHUB_TOKEN'] = token
    
    try:
        owner, repo = payload['repository']['full_name'].split('/')
        output_dir = 'output'
        llm_config = LLMConfig(
            model=os.environ.get('OPENAI_MODEL', 'gpt-4'),
            api_key=os.environ.get('OPENAI_API_KEY'),
            base_url=os.environ.get('OPENAI_API_BASE')
        )
        runtime_container_image = os.environ.get('RUNTIME_CONTAINER_IMAGE', 'ghcr.io/all-hands-ai/openhands-runtime:latest')
        prompt_template = os.environ.get('PROMPT_TEMPLATE', 'default')
        max_iterations = int(os.environ.get('MAX_ITERATIONS', '4'))
        username = os.environ.get('GITHUB_USERNAME', 'openhands')
        
        if event == 'issues' and payload['action'] == 'labeled':
            label = payload['label']['name']
            if label == 'fix-me':
                issue_number = payload['issue']['number']
                
                # Attempt to resolve the issue
                await resolve_issue(
                    owner=owner,
                    repo=repo,
                    token=token,
                    username=username,
                    max_iterations=max_iterations,
                    output_dir=output_dir,
                    llm_config=llm_config,
                    runtime_container_image=runtime_container_image,
                    prompt_template=prompt_template,
                    issue_type='issue',
                    repo_instruction=None,
                    issue_number=issue_number
                )
                
                # Check if resolution was successful
                with open('output/output.jsonl', 'r') as f:
                    last_line = f.readlines()[-1]
                    result = ResolverOutput.model_validate_json(last_line)
                    if result.success:
                        # Create draft PR
                        send_pull_request(
                            github_issue=result.issue,
                            github_token=token,
                            github_username=username,
                            patch_dir=os.path.join(output_dir, 'patches', f'issue_{issue_number}'),
                            llm_config=llm_config,
                            pr_type='draft'
                        )
                    else:
                        # Push branch only
                        send_pull_request(
                            github_issue=result.issue,
                            github_token=token,
                            github_username=username,
                            patch_dir=os.path.join(output_dir, 'patches', f'issue_{issue_number}'),
                            llm_config=llm_config,
                            pr_type='branch',
                            additional_message=result.error
                        )
                        
        elif event == 'pull_request' and payload['action'] == 'labeled':
            label = payload['label']['name']
            if label == 'fix-me':
                pr_number = payload['pull_request']['number']
                
                # Attempt to resolve the PR
                await resolve_issue(
                    owner=owner,
                    repo=repo,
                    token=token,
                    username=username,
                    max_iterations=max_iterations,
                    output_dir=output_dir,
                    llm_config=llm_config,
                    runtime_container_image=runtime_container_image,
                    prompt_template=prompt_template,
                    issue_type='pr',
                    repo_instruction=None,
                    issue_number=pr_number
                )
                
                # Always push changes to the PR branch
                with open('output/output.jsonl', 'r') as f:
                    last_line = f.readlines()[-1]
                    result = ResolverOutput.model_validate_json(last_line)
                    send_pull_request(
                        github_issue=result.issue,
                        github_token=token,
                        github_username=username,
                        patch_dir=os.path.join(output_dir, 'patches', f'pr_{pr_number}'),
                        llm_config=llm_config,
                        pr_type='branch',
                        additional_message=result.error if not result.success else None
                    )
                
        return {'status': 'success'}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 5000))
    uvicorn.run(app, host='0.0.0.0', port=port)