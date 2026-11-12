"""Conservative token-count lower bound for the locked NFC/ByteLevel tokenizer."""
import json,re
from pathlib import Path
from tokenizers.pre_tokenizers import ByteLevel


class CapacityBound:
    def __init__(self,config):
        assert config['normalizer']=={'type':'NFC'}
        parts=config['pre_tokenizer'];assert parts['type']=='Sequence' and len(parts['pretokenizers'])==2
        split,byte=parts['pretokenizers']
        assert split['type']=='Split' and split['behavior']=='Isolated' and not split['invert']
        assert byte=={'type':'ByteLevel','add_prefix_space':False,'trim_offsets':False,'use_regex':False}
        assert config['post_processor']==byte and config['decoder']==byte and config['model']['type']=='BPE'
        added=config['added_tokens']
        assert all(not any(t[k] for k in ('single_word','lstrip','rstrip','normalized')) and t['content'] and t['content'].isascii() for t in added)
        self.added={t['content'] for t in added};alphabet=set(ByteLevel.alphabet());maximum=0
        assert alphabet<=set(config['model']['vocab']),'Byte alphabet incomplete'
        for token in config['model']['vocab']:
            if token not in self.added:
                assert set(token)<=alphabet,'Unsupported non-byte vocabulary token'
                maximum=max(maximum,len(token))
        self.maximum=max(maximum,max(len(t.encode()) for t in self.added))
        self.pattern=re.compile('('+'|'.join(re.escape(t) for t in sorted(self.added,key=len,reverse=True))+')')
    @classmethod
    def load(cls,path='.cache/references/tokenizer.json'):return cls(json.loads(Path(path).read_text()))
    def measure(self,text,normalizer):
        # AddedToken(normalized=False) matches before NFC. Isolated splitting and
        # ByteLevel preserve every payload byte; one ID covers at most maximum.
        chunks=self.pattern.split(text)
        size=sum(len((part if i%2 else normalizer.normalize_str(part)).encode('utf-8')) for i,part in enumerate(chunks))
        return (size+self.maximum-1)//self.maximum,size
