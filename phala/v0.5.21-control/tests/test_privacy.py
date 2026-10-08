import importlib.util,io,json,logging,sys
from pathlib import Path
root=Path(sys.argv[1]); package=root/'python/sglang/_phala_privacy'
spec=importlib.util.spec_from_file_location('privacy_fixture',package/'runtime.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); mod.install()
secret='fixture-secret-body-token-canary'; stream=io.StringIO(); handler=logging.StreamHandler(stream); logger=logging.getLogger('sglang.fixture'); logger.handlers=[handler]; logger.propagate=False; logger.setLevel(logging.DEBUG)
sites=json.loads((package/'privacy_sites.json').read_text())
for key in sites:
    file,line=key.rsplit(':',1)
    # LogRecordFactory is shared by all logging handlers, including non-propagating loggers.
    record=logger.makeRecord(logger.name,logging.ERROR,'/sgl-workspace/sglang/python/sglang/srt/'+file,int(line),secret+' %s',(secret,),(ValueError,ValueError(secret),None))
    logger.handle(record)
assert secret not in stream.getvalue()
assert len(stream.getvalue().splitlines())>=len(sites)
# The original argument evaluation remains intact, unlike source-call rewrites.
counter=[]
def effect(): counter.append(1); return secret
first=next(iter(sites)); file,line=first.rsplit(':',1)
logger.handle(logger.makeRecord(logger.name,logging.ERROR,'/sgl-workspace/sglang/python/sglang/srt/'+file,int(line),'%s',(effect(),),None))
assert counter==[1] and secret not in stream.getvalue()
# Benign numeric diagnostics at unmapped code locations stay available.
record=logger.makeRecord(logger.name,logging.INFO,'/sgl-workspace/sglang/python/sglang/srt/benign.py',1,'batch=%d',(4,),None); logger.handle(record)
assert 'batch=4' in stream.getvalue()
print('PASS privacy mapped sites',len(sites),'all-handler factory, exception stripping, argument side effect preserved, benign numeric diagnostics')
