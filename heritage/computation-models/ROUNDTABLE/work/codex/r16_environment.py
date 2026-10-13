import sys, importlib.util, importlib.metadata, json, os,platform
out={'python':sys.version,'cwd':os.getcwd(),'platform':platform.platform(),'modules':{m:bool(importlib.util.find_spec(m)) for m in ['sympy','numpy','psutil','requests','pypdf']},'versions':{m:importlib.metadata.version(m) for m in ['sympy','numpy','psutil','requests','pypdf']}}
json.dump(out,open('r16_ENVIRONMENT.json','w'),indent=2)
print(json.dumps(out))
