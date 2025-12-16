'''
    Tests for the recommendation microservices.

    These modules are used to test the recommendation microservices for the following scenarios:
    1. New user - no personal recs, no online events (should get popular tracks)
    2. User with events only - no personal recs, has events (should get blended popular + online)
    3. User with personal recs only - has personal recs, no events (should get personal recs)
    4. User with both - has personal recs and events (should get blended personal + online)

    Usage examples:
    python3 -m src.microservice.test_services # run all tests
'''

# ---------- Imports ---------- #
import pytest
import sys
import traceback
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.microservice.final_recs import app
from src.logging_setup import setup_logging

# ---------- Logging setup ---------- #
logger = setup_logging('test_services')

# ---------- Initialize a test client ---------- #
logger.info('Initializing test client')
client = TestClient(app)

# ---------- Test data ---------- #
# Fake recommendations that will be used in the tests

# Popular tracks (popularity-based recommendations for new users)
POPULAR_TRACKS = [101, 102, 103, 104, 105]

# Personal recommendations (ALS-based, for existing users)
PERSONAL_TRACKS = [201, 202, 203, 204, 205]

# Similar tracks (similarity-based recommendations)
SIMILAR_TRACKS = {
    'similar_track_id': [301, 302, 303, 304, 305],
    'similarity_score': [0.9, 0.8, 0.7, 0.6, 0.5]
}

# User's listening history (online events)
USER_EVENTS = [401, 402, 403]

# ---------- Helper function to create mock responses ---------- #
def create_mock_response(data):
    '''
        Create a mock HTTP response that returns the given data.

        Args:
        - data - data to return in the mock response

        Returns:
        - mock HTTP response
    '''
    mock = MagicMock()
    mock.json.return_value = data
    return mock

# ---------- TEST 1: New user without personal recs and without online events ---------- #
@patch('src.microservice.final_recs.requests.post')
def test_new_user_no_recs_no_events(mock_post):
    '''
        Scenario: A completely new user
        - No ALS-based (personal) recommendations
        - No online events (listening history)
        
        Expected: User gets popular (default) tracks
    '''
    
    logger.info('RUNNING: Test 1: New user without personal recs and without online events')
    # Setup: Define what each service should return
    def mock_service_responses(url, **kwargs):
        if '/get_recs' in url:
            # Offline service returns popular tracks (no personal recs for this user)
            return create_mock_response(POPULAR_TRACKS)
        
        elif '/get' in url:
            # Events service returns empty list (no listening history)
            return create_mock_response({'events': []})
        
        elif '/similar_tracks' in url:
            # Online service won't be called (no events)
            return create_mock_response({'similar_track_id': [], 'similarity_score': []})
    
    mock_post.side_effect = mock_service_responses
    
    # Call the recommendations endpoint
    response = client.post('/recommendations?user_id=99999&k=5')
    result = response.json()
    
    # Check: Should get popular tracks since there's no online history
    assert response.status_code == 200
    assert result['recs'] == POPULAR_TRACKS
    if result['recs'] == POPULAR_TRACKS:
        logger.info('PASSED: Test 1: New user gets popular tracks')
    else:
        logger.error('FAILED: Test 1: New user does not get popular tracks')


# ---------- TEST 2: User without personal recs, but with listening history ---------- #
@patch('src.microservice.final_recs.requests.post')
def test_user_no_recs_with_events(mock_post):
    '''
        Scenario: User without personal recommendations but has listening history
        - No ALS-based (personal) recommendations
        - Has online events (listened to some tracks)
        
        Expected: User gets blended recommendations (online + popular)
    '''
    
    logger.info('RUNNING: Test 2: User without personal recs, but with online events')
    def mock_service_responses(url, **kwargs):
        if '/get_recs' in url:
            # Offline service returns popular tracks
            return create_mock_response(POPULAR_TRACKS)
        
        elif '/get' in url:
            # Events service returns user's listening history
            return create_mock_response({'events': USER_EVENTS})
        
        elif '/similar_tracks' in url:
            # Online service returns similar tracks
            return create_mock_response(SIMILAR_TRACKS)
    
    mock_post.side_effect = mock_service_responses
    
    # Call the recommendations endpoint
    response = client.post('/recommendations?user_id=88888&k=5')
    result = response.json()
    
    # Check: Should get blended recommendations
    assert response.status_code == 200
    assert len(result['recs']) > 0
    
    # First item should be from online recs (similar tracks)
    # Second item should be from offline recs (popular tracks)
    assert result['recs'][0] == SIMILAR_TRACKS['similar_track_id'][0]  # 301
    assert result['recs'][1] == POPULAR_TRACKS[0]  # 101

    if result['recs'][0] == SIMILAR_TRACKS['similar_track_id'][0] and result['recs'][1] == POPULAR_TRACKS[0]:
        logger.info('PASSED: Test 2: User with events gets blended (online + popular) recommendations')
    else:
        logger.error('FAILED: Test 2: User with events does not get blended (online + popular) recommendations')


# ---------- TEST 3: User with personal recs, but without online events ---------- #
@patch('src.microservice.final_recs.requests.post')
def test_user_with_recs_no_events(mock_post):
    '''
        Scenario: User with personal recommendations but no recent activity
        - Has ALS-based (personal) recommendations
        - No online events (no recent listening history)
        
        Expected: User gets personal recommendations only
    '''
    
    logger.info('RUNNING: Test 3: User with personal recs, but without online events')
    def mock_service_responses(url, **kwargs):
        if '/get_recs' in url:
            # Offline service returns personal tracks for this user
            return create_mock_response(PERSONAL_TRACKS)
        
        elif '/get' in url:
            # Events service returns empty list
            return create_mock_response({'events': []})
        
        elif '/similar_tracks' in url:
            # Online service won't be called (no events)
            return create_mock_response({'similar_track_id': [], 'similarity_score': []})
    
    mock_post.side_effect = mock_service_responses
    
    # Call the recommendations endpoint
    response = client.post('/recommendations?user_id=77777&k=5')
    result = response.json()
    
    # Check: Should get personal recommendations only
    assert response.status_code == 200
    assert result['recs'] == PERSONAL_TRACKS

    if result['recs'] == PERSONAL_TRACKS:
        logger.info('PASSED: Test 3: User with personal recs (no events) gets personal recommendations')
    else:
        logger.error('FAILED: Test 3: User with personal recs (no events) does not get personal recommendations')


# ---------- TEST 4: User with both personal recs and online events ---------- #
@patch('src.microservice.final_recs.requests.post')
def test_user_with_both_recs_and_events(mock_post):
    '''
    Scenario: Active user with history and personal recommendations
    - Has ALS-based (personal) recommendations
    - Has online events (recent listening history)
    
    Expected: User gets blended recommendations (online + personal)
    Blending pattern: online[0], personal[0], online[1], personal[1], ...
    '''
    
    logger.info('RUNNING: Test 4: User with both personal recs and online events')
    def mock_service_responses(url, **kwargs):
        if '/get_recs' in url:
            # Offline service returns personal tracks
            return create_mock_response(PERSONAL_TRACKS)
        
        elif '/get' in url:
            # Events service returns user's listening history
            return create_mock_response({'events': USER_EVENTS})
        
        elif '/similar_tracks' in url:
            # Online service returns similar tracks
            return create_mock_response(SIMILAR_TRACKS)
    
    mock_post.side_effect = mock_service_responses
    
    # Call the recommendations endpoint
    response = client.post('/recommendations?user_id=66666&k=6')
    result = response.json()
    
    # Check: Should get blended recommendations
    assert response.status_code == 200
    
    # Verify blending pattern: alternating online and personal
    # Position 0: online rec (301)
    # Position 1: personal rec (201)
    # Position 2: online rec (302)
    # Position 3: personal rec (202)
    assert result['recs'][0] == SIMILAR_TRACKS['similar_track_id'][0]  # 301
    assert result['recs'][1] == PERSONAL_TRACKS[0]  # 201

    if result['recs'][0] == SIMILAR_TRACKS['similar_track_id'][0] and result['recs'][1] == PERSONAL_TRACKS[0]:
        logger.info('PASSED: Test 4: User with both recs and events gets blended recommendations')
    else:
        logger.error('FAILED: Test 4: User with both recs and events does not get blended recommendations')

# ---------- Run all tests ---------- #
if __name__ == '__main__':
    logger.info('Running recommendation service tests')
    # Run the tests
    try:
        pytest.main([__file__, '-v'])
        logger.info('DONE: Recommendation service tests completed successfully')
    except Exception as e:
        logger.error(f'ERROR: Recommendation service tests failed: {e}')
        traceback.print_exc()
        sys.exit(1)
