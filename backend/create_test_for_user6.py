from server import app, db, User, Document, Extraction
import json
from datetime import datetime

with app.app_context():
    # Check if user 6 exists
    user = User.query.get(6)
    if not user:
        print("User with ID 6 not found!")
        exit(1)
    
    print(f"Found user: {user.name} ({user.email})")
    
    # Check if user has any documents
    documents = Document.query.filter_by(user_id=6).all()
    
    document_id = None
    if documents:
        document_id = documents[0].id
        print(f"Using existing document with ID: {document_id}")
    else:
        # Create a test document for this user
        test_document = Document(
            filename="test_document_user6.pdf",
            file_hash="test_hash_user6_123",
            file_type="pdf",
            user_id=6
        )
        db.session.add(test_document)
        db.session.flush()  # Get the ID without committing
        document_id = test_document.id
        print(f"Created test document with ID: {document_id}")
    
    # Create test extraction
    test_extraction = Extraction(
        content={
            "entities": ["Company A", "Company B"],
            "dates": ["2023-01-01", "2024-12-31"],
            "scope": "Software development services including design, implementation, and maintenance.",
            "sla": "99.9% uptime guarantee with response time within 2 hours for critical issues.",
            "penalties": "2% of monthly fee for each 0.1% below SLA, capped at 20% of monthly fees.",
            "confidentiality": "All information shared is confidential for 5 years after termination.",
            "termination": "30 days written notice required for termination without cause.",
            "commercials": "Fixed monthly fee of $25,000 with annual increase of 3%.",
            "risks": ["Ambiguous service definition", "Limited liability cap"]
        },
        document_id=document_id,
        user_id=6
    )
    
    db.session.add(test_extraction)
    db.session.commit()
    
    print(f"Test extraction created successfully with ID: {test_extraction.id}")
    
    # Verify extraction was created
    extractions = Extraction.query.filter_by(user_id=6).all()
    print(f"Found {len(extractions)} extractions for user 6:")
    for e in extractions:
        print(f"Extraction ID: {e.id}, Document ID: {e.document_id}, Created at: {e.created_at}") 