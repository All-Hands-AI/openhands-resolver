from typing import Any, Literal
import json
import logging
import requests
from litellm import BaseModel

logger = logging.getLogger(__name__)

class FeedbackDataModel(BaseModel):
    version: str
    email: str
    token: str
    feedback: Literal['positive', 'negative']
    permissions: Literal['public', 'private']
    trajectory: list[dict[str, Any]]


FEEDBACK_URL = 'https://share-od-trajectory-3u9bw9tx.uc.gateway.dev/share_od_trajectory'
VIEWER_PAGE = "https://www.all-hands.dev/share"


def store_feedback(feedback: FeedbackDataModel) -> dict[str, str]:
    # Start logging
    display_feedback = feedback.model_dump()
    if 'trajectory' in display_feedback:
        display_feedback['trajectory'] = (
            f"elided [length: {len(display_feedback['trajectory'])}"
        )
    if 'token' in display_feedback:
        display_feedback['token'] = 'elided'
    logger.info(f'Got feedback: {display_feedback}')
    
    # Start actual request
    response = requests.post(
        FEEDBACK_URL,
        headers={'Content-Type': 'application/json'},
        json=feedback.model_dump(),
    )
    if response.status_code != 200:
        raise ValueError(f'Failed to store feedback: {response.text}')
    response_data = json.loads(response.text)
    logger.info(f'Stored feedback: {response.text}')
    return response_data


def get_trajectory_url(feedback_id: str) -> str:
    """Get the URL to view the trajectory."""
    return f"{VIEWER_PAGE}?share_id={feedback_id}"


def submit_resolver_output(output: 'ResolverOutput', token: str) -> str:
    """Submit a ResolverOutput to Share-OpenHands and return the trajectory URL."""
    feedback = FeedbackDataModel(
        version='1.0',
        email='openhands@all-hands.dev',
        token=token,
        feedback='positive' if output.success else 'negative',
        permissions='private',
        trajectory=output.history
    )
    
    response_data = store_feedback(feedback)
    trajectory_url = get_trajectory_url(response_data['feedback_id'])
    return trajectory_url
