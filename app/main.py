import os
import uvicorn
from dotenv import load_dotenv
from app.app import create_app

app = create_app()
load_dotenv()

def main():
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=os.getenv("PORT", 33001),
        log_level="info",
        reload=True,      # only in dev
    )

if __name__ == "__main__":
    main()
