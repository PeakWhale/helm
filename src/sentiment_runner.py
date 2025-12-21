
import sys
import json
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR, stream=sys.stderr)

def main():
    # Load model ONCE at startup
    try:
        sys.stderr.write("[Sentiment Runner] Loading FinBERT model... (this may take a moment)\n")
        sys.stderr.flush()
        
        from transformers import pipeline
        pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        
        sys.stderr.write("[Sentiment Runner] FinBERT model loaded and ready.\n")
        sys.stderr.flush()
        
        # Signal readiness
        print(json.dumps({"status": "ready"}), flush=True)
    except Exception as e:
        print(json.dumps({"error": f"Model load failed: {str(e)}"}), flush=True)
        return

    # Loop forever reading requests
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
            
        try:
            # Expecting just raw text input per line, or JSON?
            # Let's assume JSON input for better structure: {"text": "..."}
            req = json.loads(line)
            text = req.get("text", "")
            
            result = pipe(text[:512])[0]
            output = {
                "label": result.get("label", "UNKNOWN"),
                "score": result.get("score", 0.0)
            }
            print(json.dumps(output), flush=True)
            
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)

if __name__ == "__main__":
    main()
