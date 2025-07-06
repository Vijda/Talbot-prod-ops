import requests
import json

def test_ollama_connection():
    """Test if Ollama is running and accessible."""
    base_url = "http://localhost:11434"
    
    try:
        # Test basic connection
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            print("✅ Ollama is running and accessible!")
            
            # Check available models
            models = response.json().get('models', [])
            print(f"Available models: {[model['name'] for model in models]}")
            
            # Check if llama3.2 is available
            llama3_available = any('llama3.2' in model['name'] for model in models)
            if llama3_available:
                print("✅ Llama3.2 model is available!")
                return True
            else:
                print("❌ Llama3.2 model not found. Available models:")
                for model in models:
                    print(f"  - {model['name']}")
                print("\nTo install llama3.2, run: ollama pull llama3.2")
                return False
        else:
            print(f"❌ Ollama connection failed with status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama. Please make sure:")
        print("  1. Ollama is installed: https://ollama.ai/")
        print("  2. Ollama service is running")
        print("  3. The service is accessible at http://localhost:11434")
        return False
    except Exception as e:
        print(f"❌ Error testing Ollama connection: {e}")
        return False

def test_sentiment_analysis():
    """Test a simple sentiment analysis with Ollama."""
    if not test_ollama_connection():
        return False
    
    print("\n=== Testing Sentiment Analysis ===")
    
    test_text = "The food was delicious and the service was excellent!"
    
    try:
        payload = {
            "model": "llama3.2",
            "prompt": f"""
            Analyze the sentiment of the following review and classify it as POSITIVE, NEGATIVE, or NEUTRAL.
            
            Review: "{test_text}"
            
            Please respond with only the classification (POSITIVE/NEGATIVE/NEUTRAL) and a brief reason.
            Format: CLASSIFICATION: [POSITIVE/NEGATIVE/NEUTRAL] | REASON: [brief explanation]
            """,
            "stream": False
        }
        
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Sentiment analysis test successful!")
            print(f"Input: {test_text}")
            print(f"Response: {result.get('response', '').strip()}")
            return True
        else:
            print(f"❌ Sentiment analysis test failed with status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing sentiment analysis: {e}")
        return False

if __name__ == "__main__":
    print("=== Ollama Connection Test ===")
    
    # Test connection
    connection_ok = test_ollama_connection()
    
    if connection_ok:
        # Test sentiment analysis
        test_sentiment_analysis()
    
    print("\n=== Test Complete ===")
    if connection_ok:
        print("✅ You're ready to run the sentiment analysis script!")
        print("Run: python sentiment_analysis.py")
    else:
        print("❌ Please fix the Ollama setup before running the sentiment analysis.") 