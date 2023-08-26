from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# middlewares
app.add_middleware(
    CORSMiddleware, # https://fastapi.tiangolo.com/tutorial/cors/
    allow_origins=['*'], # wildcard to allow all, more here - https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Origin
    allow_credentials=True, # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Credentials
    allow_methods=['*'], # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Methods
    allow_headers=['*'], # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Headers
)

@app.get('/')
async def root():
    return {'hello': 'world'}


import nest_asyncio
from pyngrok import ngrok
import uvicorn


# specify a port
port = 8000
ngrok_tunnel = ngrok.connect(port)

# where we can visit our fastAPI app
print('Public URL:', ngrok_tunnel.public_url)


nest_asyncio.apply()

# finally run the app
uvicorn.run(app, port=port)