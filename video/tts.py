import json,sys,os,base64,urllib.request
key=[l.split('=',1)[1].strip() for l in open('.env') if l.startswith('ELEVENLABS_API_KEY')][0]
def tts(voice,text,out,model='eleven_multilingual_v2',settings=None):
    body={'text':text,'model_id':model,'voice_settings':settings or {'stability':0.4,'similarity_boost':0.8,'style':0.45,'use_speaker_boost':True}}
    req=urllib.request.Request(f'https://api.elevenlabs.io/v1/text-to-speech/{voice}/with-timestamps?output_format=mp3_44100_128',data=json.dumps(body).encode(),headers={'xi-api-key':key,'content-type':'application/json'})
    try: d=json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e: print(out,e.code,e.read()[:300]);return
    open(out+'.mp3','wb').write(base64.b64decode(d['audio_base64']))
    json.dump(d['alignment'],open(out+'.json','w'))
    print('ok',out)
if __name__=='__main__':
    tts(sys.argv[1],open(sys.argv[2]).read().strip(),sys.argv[3],*(sys.argv[4:5]))
