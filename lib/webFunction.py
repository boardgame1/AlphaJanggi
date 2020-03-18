import requests
import os
from urllib.request import urlopen, Request
from tqdm import tqdm

def upload_file(url, path, filename=None, data=None, rm=False):
    filename = filename if filename is not None else 'file'
    files = {'file': (filename, open(path, 'rb'))}
    success = False
    for i in range(3):
        try:
            r = requests.post(url, files=files, data=data)
            if r.status_code != 200:
                print(f"Error occurs when upload {filename}: {r.text}")
            else:
                success = True
                break
        except Exception as e:
            print(f"Error occurs when upload {filename}: {e}")
    if rm:
        os.remove(path)
    return True if success else None

def http_request(url, post=False, data=None):
    success = False
    for i in range(3):
        try:
            if post:
                r = requests.post(url, data=data, timeout=10)
            else:
                r = requests.get(url)
            if r.status_code != 200:
                print(f"Error occurs when request {url}: {r.text}")
            else:
                success = True
                break
        except Exception as e:
            print(f"Error occurs when request {url}: {e}")
    return r.json() if success else None

def download_file(url, save_path):
    if os.path.exists(save_path):
        os.remove(save_path)
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    file_size = int(urlopen(req).info().get('Content-Length', -1))
    first_byte = 0
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1])
    req = requests.get(url, headers=header, stream=True)
    with(open(save_path, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return True
