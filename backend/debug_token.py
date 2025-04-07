import jwt
from server import app

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NDQxODM2OTQsImlhdCI6MTc0MzU3ODg5NCwic3ViIjo2fQ.0_rNcK75iG5U6nVHaug7OpI7Jo0QFVHCXTqdRST4FSc"

with app.app_context():
    try:
        # Decode the token
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        user_id = payload['sub']
        print(f"Token valid! User ID: {user_id}")
        
        # Import User model here to avoid circular imports
        from server import User
        
        # Check if user exists
        user = User.query.get(user_id)
        if user:
            print(f"User found: {user.name} ({user.email})")
        else:
            print(f"User with ID {user_id} not found in database!")
            
        # Check test extraction
        from server import Extraction
        
        # Get extraction for this user
        extractions = Extraction.query.filter_by(user_id=user_id).all()
        print(f"Found {len(extractions)} extractions for user {user_id}")
        
        # Get extractions for all users
        all_extractions = Extraction.query.all()
        print(f"Total extractions in database: {len(all_extractions)}")
        for e in all_extractions:
            print(f"Extraction ID: {e.id}, User ID: {e.user_id}, Document ID: {e.document_id}")
        
    except Exception as e:
        print(f"Error decoding token: {e}") 