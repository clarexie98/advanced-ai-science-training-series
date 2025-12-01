"""
Quick test script to verify tools work independently before running the full multi-agent system.
Run this first to catch any dependency or API issues.
"""

import sys

def test_imports():
    """Test that all required packages are installed."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    try:
        import arxiv
        print("✅ arxiv imported successfully")
    except ImportError as e:
        print(f"❌ arxiv import failed: {e}")
        return False
    
    try:
        import pubchempy
        print("✅ pubchempy imported successfully")
    except ImportError as e:
        print(f"❌ pubchempy import failed: {e}")
        return False
    
    try:
        import requests
        print("✅ requests imported successfully")
    except ImportError as e:
        print(f"❌ requests import failed: {e}")
        return False
    
    try:
        from langchain_openai import ChatOpenAI
        print("✅ langchain_openai imported successfully")
    except ImportError as e:
        print(f"❌ langchain_openai import failed: {e}")
        return False
    
    try:
        from langgraph.graph import StateGraph
        print("✅ langgraph imported successfully")
    except ImportError as e:
        print(f"❌ langgraph import failed: {e}")
        return False
    
    print()
    return True


def test_tools():
    """Test each tool individually."""
    print("=" * 60)
    print("Testing custom tools...")
    print("=" * 60)
    
    from clarexie_tools import (
        search_arxiv,
        lookup_molecular_properties,
        calculate_simple_statistics,
        search_protein_database
    )
    
    # Test 1: arXiv search
    print("\n1. Testing search_arxiv...")
    try:
        result = search_arxiv.invoke({"query": "quantum computing", "max_results": 1})
        print(f"✅ search_arxiv works: Found {len(result)} characters of data")
    except Exception as e:
        print(f"❌ search_arxiv failed: {e}")
    
    # Test 2: Molecular properties
    print("\n2. Testing lookup_molecular_properties...")
    try:
        result = lookup_molecular_properties.invoke({"molecule_name": "water"})
        print(f"✅ lookup_molecular_properties works: {result[:100]}...")
    except Exception as e:
        print(f"❌ lookup_molecular_properties failed: {e}")
    
    # Test 3: Statistics
    print("\n3. Testing calculate_simple_statistics...")
    try:
        result = calculate_simple_statistics.invoke({
            "numbers": [1.0, 2.0, 3.0, 4.0, 5.0],
            "operation": "all"
        })
        print(f"✅ calculate_simple_statistics works: {result[:100]}...")
    except Exception as e:
        print(f"❌ calculate_simple_statistics failed: {e}")
    
    # Test 4: Protein search
    print("\n4. Testing search_protein_database...")
    try:
        result = search_protein_database.invoke({"protein_name": "insulin"})
        print(f"✅ search_protein_database works: {result[:100]}...")
    except Exception as e:
        print(f"❌ search_protein_database failed: {e}")
    
    print()


def test_authentication():
    """Test ALCF authentication."""
    print("=" * 60)
    print("Testing ALCF authentication...")
    print("=" * 60)
    
    try:
        from inference_auth_token import get_access_token
        token = get_access_token()
        print(f"✅ Authentication successful: Token retrieved ({len(token)} characters)")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("   Try running: python inference_auth_token.py --force authenticate")
        return False
    
    print()
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🧪 Pre-flight Test Suite for Clare's Multi-Agent System")
    print("=" * 60 + "\n")
    
    success = True
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed. Run: pip install -r requirements.txt")
        success = False
    
    # Test 2: Authentication
    if not test_authentication():
        print("\n❌ Authentication test failed.")
        success = False
    
    # Test 3: Tools
    try:
        test_tools()
    except Exception as e:
        print(f"\n❌ Tool test failed with error: {e}")
        success = False
    
    # Summary
    print("=" * 60)
    if success:
        print("✅ All tests passed! You're ready to run:")
        print("   python clarexie_multi_agent.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        sys.exit(1)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
