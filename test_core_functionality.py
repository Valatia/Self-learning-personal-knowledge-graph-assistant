#!/usr/bin/env python3
"""
Test script for core functionality without spaCy dependency.
"""

import sys
import os
sys.path.append('src')

from datetime import datetime
from rexi.models.entities import Entity, EntityType
from rexi.models.relationships import Relationship, RelationshipType
from rexi.services.neo4j_service import Neo4jService

def test_models():
    """Test basic model functionality."""
    print("Testing models...")
    
    # Test Entity creation
    entity = Entity(
        name="Test Entity",
        type=EntityType.CONCEPT,
        description="A test concept entity",
        confidence=0.9
    )
    print(f"✓ Entity created: {entity.name} ({entity.type})")
    
    # Test Relationship creation
    relationship = Relationship(
        source_entity_id="1",
        target_entity_id="2", 
        type=RelationshipType.RELATED_TO,
        confidence=0.8
    )
    print(f"✓ Relationship created: {relationship.type}")
    
    return True

def test_neo4j_service():
    """Test Neo4j service initialization."""
    print("\nTesting Neo4j service...")
    
    try:
        service = Neo4jService()
        print("✓ Neo4j service initialized successfully")
        
        # Test method availability
        methods = [
            'get_all_entities', 'get_all_relationships', 'get_old_entities',
            'get_temporal_relationships', 'batch_update_nodes', 'batch_delete_nodes',
            'get_entities_by_type', 'find_entities_with_temporal_validity',
            'create_temporal_relationship', 'update_relationship'
        ]
        
        for method in methods:
            if hasattr(service, method):
                print(f"✓ Method {method} available")
            else:
                print(f"✗ Method {method} missing")
        
        return True
        
    except Exception as e:
        print(f"✗ Neo4j service failed: {e}")
        return False

def test_memory_evolution_imports():
    """Test memory evolution imports without spaCy."""
    print("\nTesting memory evolution...")
    
    try:
        # Test basic imports
        from rexi.agents.memory_evolution import MemoryEvolutionEngine
        print("✓ MemoryEvolutionEngine imported")
        
        from rexi.agents.temporal_reasoning import TemporalReasoningEngine
        print("✓ TemporalReasoningEngine imported")
        
        from rexi.agents.entity_resolver import EntityResolver
        print("✓ EntityResolver imported")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== REXI Core Functionality Test ===\n")
    
    results = []
    results.append(test_models())
    results.append(test_neo4j_service())
    results.append(test_memory_evolution_imports())
    
    print(f"\n=== Test Results ===")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All core functionality tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
