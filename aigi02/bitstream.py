"""Compact, versioned and bounded bitstream framing (no pickle).

V2 fixed fields are 50 bytes, six unsigned varint lengths follow, then payloads
and an eight-byte SHA-256 integrity suffix. The minimum overhead is 64 bytes.
Model/runtime identity is a 128-bit profile key; text and prompt fingerprints
are 64 bits. These detect mismatched deployments, not provide authentication.
"""
from __future__ import annotations
import hashlib
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

MAGIC=b'AIG2';VERSION=2;MAX_BYTES=512*1024*1024;MAX_PIXELS=100_000_000
FIXED=struct.Struct('>4sBBHHIf16s8s8s')


def bounded_uncompress(data:bytes,limit:int)->bytes:
    obj=zlib.decompressobj();result=obj.decompress(data,limit+1)
    if len(result)>limit or obj.unconsumed_tail or not obj.eof or obj.unused_data:
        raise ValueError('Invalid or oversized compressed payload')
    return result


def profile_key(header):
    if 'model_key' in header:return header['model_key']
    fields=[header['architecture'],header['checkpoint'],header['entropy_profile']]
    return hashlib.sha256('|'.join(fields).encode()).hexdigest()[:32]


def varint(number):
    if type(number) is not int or not 0<=number<=MAX_BYTES:raise ValueError('Invalid segment size')
    out=bytearray()
    while number>=128:out.append((number&127)|128);number>>=7
    out.append(number);return bytes(out)


def read_varint(data,pos):
    value=0
    for shift in range(0,35,7):
        if pos>=len(data):raise ValueError('Truncated segment table')
        byte=data[pos];pos+=1;value|=(byte&127)<<shift
        if not byte&128:
            if value>MAX_BYTES:raise ValueError('Segment too large')
            return value,pos
    raise ValueError('Invalid length integer')


@dataclass
class Packet:
    header:dict
    segments:list[bytes]

    def pack(self)->bytes:
        h=self.header
        if len(self.segments)!=6:raise ValueError('Exactly six segments are required')
        height,width=h['original_size'];seed=h['seed'];lam=float(h['lambda_rd'])
        if not (0<height<=65535 and 0<width<=65535 and height*width<=MAX_PIXELS):raise ValueError('Invalid image dimensions')
        if not (0<=seed<2**32 and math.isfinite(lam) and lam>0):raise ValueError('Invalid seed or operating point')
        if h['backend'] not in ('tiny','sana') or h['prompt_mode'] not in ('free','included'):raise ValueError('Unsupported profile')
        flags=(h['backend']=='tiny')|((h['prompt_mode']=='free')<<1)|((h.get('map_source')=='oracle')<<2)
        fixed=FIXED.pack(MAGIC,VERSION,flags,height,width,seed,lam,
                         bytes.fromhex(profile_key(h)),bytes.fromhex(h['text_features_sha256'][:16]),
                         bytes.fromhex(h['prompt_sha256'][:16]))
        sizes=[len(v) for v in self.segments]
        if sum(sizes)>MAX_BYTES:raise ValueError('Packet too large')
        body=fixed+b''.join(varint(n) for n in sizes)+b''.join(self.segments)
        return body+hashlib.sha256(body).digest()[:8]

    @classmethod
    def unpack(cls,data:bytes)->'Packet':
        if len(data)<FIXED.size+6+8 or len(data)>MAX_BYTES+128:raise ValueError('Invalid packet size')
        if hashlib.sha256(data[:-8]).digest()[:8]!=data[-8:]:raise ValueError('Packet checksum mismatch')
        magic,version,flags,h,w,seed,lam,model,text,prompt=FIXED.unpack(data[:FIXED.size])
        if magic!=MAGIC or version!=VERSION or flags&~7:raise ValueError('Unsupported bitstream version')
        if not (h>0 and w>0 and h*w<=MAX_PIXELS and math.isfinite(lam) and lam>0):raise ValueError('Invalid image or rate')
        pos=FIXED.size;lengths=[]
        for _ in range(6):
            size,pos=read_varint(data,pos);lengths.append(size)
        if sum(lengths)!=len(data)-pos-8:raise ValueError('Segment sizes do not match file')
        segments=[]
        for size in lengths:segments.append(data[pos:pos+size]);pos+=size
        if flags&2 and lengths[0]!=0:raise ValueError('Free-prompt profile must not carry prompt bytes')
        header={'original_size':[h,w],'seed':seed,'lambda_rd':lam,'model_key':model.hex(),
                'text_features_sha256':text.hex(),'prompt_sha256':prompt.hex(),
                'backend':'tiny' if flags&1 else 'sana','prompt_mode':'free' if flags&2 else 'included',
                'map_source':'oracle' if flags&4 else 'distilled'}
        return cls(header,segments)

    @classmethod
    def read(cls,path):
        path=Path(path)
        if path.stat().st_size>MAX_BYTES+128:raise ValueError('Input is too large')
        return cls.unpack(path.read_bytes())

    def write(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(self.pack());tmp.replace(path)

    def prompt(self,external=None):
        if self.header['prompt_mode']=='included':prompt=bounded_uncompress(self.segments[0],1_000_000).decode('utf-8')
        elif external is not None:prompt=external
        else:raise ValueError('This stream requires the exact external prompt')
        if hashlib.sha256(prompt.encode()).hexdigest()[:16]!=self.header['prompt_sha256'][:16]:
            raise ValueError('Prompt mismatch: entropy decoding would be invalid')
        return prompt

    def rates(self):
        pixels=self.header['original_size'][0]*self.header['original_size'][1]
        total=len(self.pack());text=len(self.segments[0]);hyper=len(self.segments[1]);main=sum(map(len,self.segments[2:]))
        return {'bytes_total':total,'bytes_prompt':text,'bytes_z':hyper,'bytes_y':main,
                'bytes_header_and_integrity':total-text-hyper-main,'bpp_total':8*total/pixels,
                'bpp_visual_payload':8*(hyper+main)/pixels,'bpp_prompt_payload':8*text/pixels,
                'bpp_excluding_prompt_payload':8*(total-text)/pixels}
