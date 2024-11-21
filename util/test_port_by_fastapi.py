from typing import Union
from fastapi import FastAPI
import uvicorn

FASTAPI_PORT = 7760
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "(LAN) World on port %d" % (FASTAPI_PORT, )}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


if '__main__' == __name__:

    # 备忘：用uvicorn启动的命令行
    # uvicorn audio_chart_server_redis_http_server:app --host 0.0.0.0 --port 7761

    # https://stackoverflow.com/questions/63177681/is-there-a-difference-between-running-fastapi-from-uvicorn-command-in-dockerfile
    uvicorn.run('test_port_by_fastapi:app', host='0.0.0.0', port=FASTAPI_PORT)
