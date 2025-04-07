from server import app, db, User, Document, Extraction
import json
from datetime import datetime

# Create a context to ensure the database is initialized
with app.app_context():
    # Check if documents exist
    documents = Document.query.all()
    if not documents:
        print("No documents found. Creating a test document...")
        # Find a user
        user = User.query.first()
        if not user:
            print("No users found. Please create a user first.")
            exit(1)
        
        # Create a test document
        test_doc = Document(
            filename="test_document.pdf",
            file_hash="test_hash_123",
            file_type="pdf",
            user_id=user.id
        )
        db.session.add(test_doc)
        db.session.commit()
        doc_id = test_doc.id
        print(f"Created test document with ID: {doc_id}")
    else:
        doc_id = documents[0].id
        print(f"Using existing document with ID: {doc_id}")
    
    # Find a user
    user = User.query.first()
    if not user:
        print("No users found. Please create a user first.")
        exit(1)
    
    # Create a test extraction
    content = {
        "entities": "**Company A** and **Company B**",
        "dates": "**January 1, 2023** to **December 31, 2023**",
        "scope": "Consulting services",
        "sla": "99.9% uptime",
        "penalty": "Penalties for late delivery",
        "confidentiality": "All data is confidential",
        "termination": "30 days notice required",
        "commercials": "$10,000 per month",
        "risks": "No identified risks"
    }
    
    test_extraction = Extraction(
        content=content,
        document_id=doc_id,
        user_id=user.id
    )
    
    db.session.add(test_extraction)
    db.session.commit()
    
    print(f"Created test extraction with ID: {test_extraction.id}")
    
    # Verify extractions are in the database
    extractions = Extraction.query.filter_by(user_id=user.id).all()
    print(f"Found {len(extractions)} extractions for user {user.id}")
    for e in extractions:
        print(f"Extraction ID: {e.id}, Document ID: {e.document_id}, Created: {e.created_at}") 