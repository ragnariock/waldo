# Application

Web app using Flask and Google Cloud for demonstration purposes.

## Requirements
- Inference model
- Google Cloud

## Input Parameters
- Buffer: Maximum number of frames allowed between two bounding boxes for smoothing. (Use 0 for no smoothing.)
- Confidence: Minimum detection score allowed.

In the case of an image, a buffer of 0 indicates no additional suppression of overlapping bounding boxes. All other values mean yes.

## API Call
If the app is running, a request can be sent using the following:
```
def send_request(url, conf, buff, file):
   
    import requests
    import magic
    import json
    import os
       
    mime = magic.Magic(mime=True)
    payload =  {'conf':conf,'buff':buff}
   
    f = open(file, 'rb')
   
    files = {
     'json': ('json', json.dumps(payload), 'application/json'),
     'file': (os.path.basename(file), f, mime.from_file(file))
    }
   
    r = requests.post(url, files=files)
    f.close()
    print(str(r.content, 'utf-8'))

send_request( 'IP:8050/api', conf, buff, file)
```
