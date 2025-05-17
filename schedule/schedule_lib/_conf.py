from typing import Literal

APPS = ["memcache", "mysql", "keydb", "redis", "nginx"]
App = Literal["memcache", "mysql", "keydb", "redis", "nginx"]

COLOCATE_DATA_NUM = 4

QOS_THRESHOLD = 0.8
