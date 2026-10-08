"""Build-bound log-site redaction. No request processing or log arguments change."""
import json
import logging
from pathlib import Path

_SITES = json.loads(Path(__file__).with_name('privacy_sites.json').read_text())

def install():
    old = logging.getLogRecordFactory()
    if getattr(old, '_phala_privacy', False):
        return

    def factory(*args, **kwargs):
        record = old(*args, **kwargs)
        path = record.pathname.replace('\\', '/')
        marker = '/sglang/srt/'
        if marker in path:
            key = path.split(marker, 1)[1] + ':' + str(record.lineno)
            template = _SITES.get(key)
            if template is not None:
                record.msg = template
                record.args = ()
                record.exc_info = None
                record.exc_text = None
                record.stack_info = None
                record.__dict__.pop('color_message', None)
        return record

    factory._phala_privacy = True
    logging.setLogRecordFactory(factory)

